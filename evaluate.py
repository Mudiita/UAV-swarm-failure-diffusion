# seen vs. unseen evaluation for a trained model -- reports assignment
# accuracy, trajectory MSE, recovery success rate and the generalization gap G
import sys, os, argparse, json, re
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from sim.env import D_STATE, D_OUT, N_UAV, H, WORLD_SIZE
from models.nets import StateEncoder, FactorizedConditionEncoder, MonolithicConditionEncoder, DiffusionDenoiser, MLPBaseline
from models.diffusion_utils import DDPMScheduler

SUCCESS_TRAJ_THRESHOLD = 0.05  # normalized-coordinate MSE threshold for "successful" trajectory

# --- diagnostic-metric thresholds. All positions here are normalized by
# WORLD_SIZE (as in target_vector), so e.g. 0.05 == 5 world units.
# There is no physical range/collision model in sim/env.py, so these are
# reporting-time assumptions, sized against this benchmark's own scale
# (start/task positions are ~uniform over an 80x80 area, so typical
# start->task distances run up to ~1.13 normalized units).
COLLISION_SAFETY_DIST = 0.05          # min pairwise UAV separation below this is flagged as a collision risk
TASK_COMPLETION_TOL = 0.05            # endpoint distance tolerance vs. the reference trajectory's endpoint
BATTERY_RANGE_PER_FULL_CHARGE = 1.5   # normalized path length a fully-charged (battery=1.0) UAV can cover

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_split(path):
    d = np.load(path, allow_pickle=True)
    return torch.tensor(d["S"]), torch.tensor(d["C"]), torch.tensor(d["Y"]), d["combo_id"]

def diagnostic_metrics(traj_pred, traj_true, S):
    """traj_pred, traj_true: (B, N_UAV, H, 2) normalized by WORLD_SIZE.
    S: (B, N_UAV, D_STATE) with columns [x, y, battery, comm, ...].
    Returns (collision_rate, battery_feasibility, task_completion), each the
    fraction of (sample[, UAV]) instances passing that diagnostic."""
    B, N = traj_pred.shape[0], traj_pred.shape[1]

    # 1. collision_rate: fraction of samples whose minimum pairwise UAV
    #    separation, over all timesteps, drops below the safety distance
    diffs = traj_pred[:, :, None, :, :] - traj_pred[:, None, :, :, :]   # (B,N,N,H,2)
    dists = torch.linalg.norm(diffs, dim=-1)                            # (B,N,N,H)
    eye = torch.eye(N, dtype=torch.bool, device=traj_pred.device)
    dists = dists.masked_fill(eye[None, :, :, None], float("inf"))      # ignore self-distance
    min_dist_per_sample = dists.reshape(B, -1).min(dim=1).values        # (B,)
    collision_rate = (min_dist_per_sample < COLLISION_SAFETY_DIST).float().mean().item()

    # 2. battery_feasibility: fraction of (sample, UAV) whose predicted path
    #    length is within their battery-derived range cap
    start = S[:, :, 0:2]                                                 # (B,N,2) normalized start pos
    battery = S[:, :, 2]                                                 # (B,N)
    full_path = torch.cat([start.unsqueeze(2), traj_pred], dim=2)        # (B,N,H+1,2)
    seg_len = torch.linalg.norm(full_path[:, :, 1:] - full_path[:, :, :-1], dim=-1)  # (B,N,H)
    path_len = seg_len.sum(dim=-1)                                       # (B,N)
    max_range = battery * BATTERY_RANGE_PER_FULL_CHARGE
    battery_feasibility = (path_len <= max_range).float().mean().item()

    # 3. task_completion: fraction of (sample, UAV) whose predicted trajectory
    #    ends near the reference trajectory's endpoint (the best available
    #    proxy for "its target position": raw task positions aren't saved in
    #    the dataset, only the reference recovery trajectory that reaches them)
    end_err = torch.linalg.norm(traj_pred[:, :, -1, :] - traj_true[:, :, -1, :], dim=-1)  # (B,N)
    task_completion = (end_err < TASK_COMPLETION_TOL).float().mean().item()

    return collision_rate, battery_feasibility, task_completion

def decode_and_score(pred, target, S):
    """pred, target: (B, D_OUT). S: (B, N_UAV, D_STATE).
    Split pred/target into assignment logits + trajectory, score both, plus
    the collision/battery/task-completion diagnostics."""
    n_assign = N_UAV * (N_UAV + 1)
    assign_pred = pred[:, :n_assign].reshape(-1, N_UAV, N_UAV + 1)
    assign_true = target[:, :n_assign].reshape(-1, N_UAV, N_UAV + 1)
    traj_pred = pred[:, n_assign:].reshape(-1, N_UAV, H, 2)
    traj_true = target[:, n_assign:].reshape(-1, N_UAV, H, 2)

    assign_correct = (assign_pred.argmax(-1) == assign_true.argmax(-1)).float().mean(dim=1)  # per-sample avg over UAVs
    traj_mse = ((traj_pred - traj_true) ** 2).mean(dim=(1, 2, 3))

    success = ((assign_correct > 0.99) & (traj_mse < SUCCESS_TRAJ_THRESHOLD)).float()

    collision_rate, battery_feasibility, task_completion = diagnostic_metrics(traj_pred, traj_true, S)

    return {
        "assign_acc": assign_correct.mean().item(),
        "traj_mse": traj_mse.mean().item(),
        "success_rate": success.mean().item(),
        "collision_rate": collision_rate,
        "battery_feasibility": battery_feasibility,
        "task_completion": task_completion,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["A", "B", "C", "D"], required=True,
                    help="architecture to build (must match the checkpoint)")
    ap.add_argument("--checkpoint", default=None,
                    help="path to a specific checkpoint .pt file (e.g. results/model_C_seed1.pt); "
                         "defaults to results/model_{model}_seed0.pt")
    ap.add_argument("--split", choices=["train", "val_unseen", "test_unseen"], default="test_unseen",
                    help="unseen split to evaluate against seen (train) performance; "
                         "use val_unseen for hyperparameter tuning and test_unseen only for final reporting")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat diffusion sampling this many times and average "
                         "(sampling starts from random noise, so B/C vary run to run; "
                         "Model A is deterministic so --runs has no effect on it)")
    ap.add_argument("--out_tag", default=None,
                    help="explicit tag for the output filename (results/eval_{model}_{out_tag}_{split}.json), "
                         "overriding the seed auto-detected from --checkpoint's basename; used by "
                         "orchestration scripts so the output name always reflects the full run "
                         "identity (e.g. dropout_p), not just the seed number")
    args = ap.parse_args()

    checkpoint_path = args.checkpoint or os.path.join("results", f"model_{args.model}_seed0.pt")

    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    train_args = ckpt["args"]

    S_tr, C_tr, Y_tr, combo_tr = load_split(os.path.join("data", "train.npz"))
    S_te, C_te, Y_te, combo_te = load_split(os.path.join("data", f"{args.split}.npz"))

    # build the model(s) once, reused across runs
    if args.model == "A":
        model = MLPBaseline(D_STATE, N_UAV, 4, D_OUT).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
    else:
        state_enc = StateEncoder(D_STATE).to(device)
        state_enc.load_state_dict(ckpt["state_enc"])
        if args.model in ("C", "D"):
            cond_enc = FactorizedConditionEncoder(dropout_p=train_args["dropout_p"]).to(device)
        else:
            cond_enc = MonolithicConditionEncoder().to(device)
        cond_enc.load_state_dict(ckpt["cond_enc"])
        denoiser = DiffusionDenoiser(D_OUT, 64, cond_enc.out_dim).to(device)
        denoiser.load_state_dict(ckpt["denoiser"])
        state_enc.eval(); cond_enc.eval(); denoiser.eval()
        scheduler = DDPMScheduler(T=train_args["T"], device=device)

    def run_once(S, C, Y, combo):
        S, C, Y = S.to(device), C.to(device), Y.to(device)
        if args.model == "A":
            with torch.no_grad():
                pred = model(S, C)
        else:
            with torch.no_grad():
                state_ctx = state_enc(S)
                cond_ctx = cond_enc(C, training=False)  # no dropout at inference
                pred = scheduler.sample(denoiser, state_ctx, cond_ctx, Y.shape, device)

        overall = decode_and_score(pred.cpu(), Y.cpu(), S.cpu())
        per_combo = {}
        for c in np.unique(combo):
            idx = torch.tensor(combo == c)
            per_combo[c] = decode_and_score(pred.cpu()[idx], Y.cpu()[idx], S.cpu()[idx])
        overall["per_combo"] = per_combo
        return overall

    n_runs = args.runs if args.model != "A" else 1  # Model A is deterministic, repeating wastes time

    results = {}
    for split_name, S, C, Y, combo in [("seen", S_tr, C_tr, Y_tr, combo_tr), ("unseen", S_te, C_te, Y_te, combo_te)]:
        run_outs = [run_once(S, C, Y, combo) for _ in range(n_runs)]

        def agg(key):
            vals = [r[key] for r in run_outs]
            return float(np.mean(vals)), float(np.std(vals))

        mean_acc, std_acc = agg("assign_acc")
        mean_mse, std_mse = agg("traj_mse")
        mean_succ, std_succ = agg("success_rate")
        mean_coll, std_coll = agg("collision_rate")
        mean_batt, std_batt = agg("battery_feasibility")
        mean_task, std_task = agg("task_completion")

        metric_keys = ["assign_acc", "traj_mse", "success_rate",
                       "collision_rate", "battery_feasibility", "task_completion"]

        # average per-combo metrics across runs too
        per_combo_avg = {}
        for c in run_outs[0]["per_combo"]:
            per_combo_avg[c] = {
                k: float(np.mean([r["per_combo"][c][k] for r in run_outs])) for k in metric_keys
            }

        results[split_name] = {
            "assign_acc": mean_acc, "assign_acc_std": std_acc,
            "traj_mse": mean_mse, "traj_mse_std": std_mse,
            "success_rate": mean_succ, "success_rate_std": std_succ,
            "collision_rate": mean_coll, "collision_rate_std": std_coll,
            "battery_feasibility": mean_batt, "battery_feasibility_std": std_batt,
            "task_completion": mean_task, "task_completion_std": std_task,
            "per_combo": per_combo_avg,
        }
        if n_runs > 1:
            print(f"[Model {args.model}] {split_name:6s} -> assign_acc={mean_acc:.3f}+-{std_acc:.3f}  "
                  f"traj_mse={mean_mse:.4f}+-{std_mse:.4f}  success={mean_succ:.3f}+-{std_succ:.3f}  "
                  f"collision={mean_coll:.3f}+-{std_coll:.3f}  battery_feas={mean_batt:.3f}+-{std_batt:.3f}  "
                  f"task_completion={mean_task:.3f}+-{std_task:.3f}  ({n_runs} runs)")
        else:
            print(f"[Model {args.model}] {split_name:6s} -> assign_acc={mean_acc:.3f}  traj_mse={mean_mse:.4f}  success={mean_succ:.3f}  "
                  f"collision={mean_coll:.3f}  battery_feas={mean_batt:.3f}  task_completion={mean_task:.3f}")

    results["generalization_gap"] = results["seen"]["success_rate"] - results["unseen"]["success_rate"]
    print(f"[Model {args.model}] Generalization gap G = {results['generalization_gap']:.3f}")

    if args.out_tag:
        seed_tag = args.out_tag
    else:
        seed_match = re.search(r"seed(\d+)", os.path.basename(checkpoint_path))
        seed_tag = f"seed{seed_match.group(1)}" if seed_match else os.path.splitext(os.path.basename(checkpoint_path))[0]

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", f"eval_{args.model}_{seed_tag}_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()
