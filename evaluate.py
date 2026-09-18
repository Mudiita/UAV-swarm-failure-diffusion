# seen vs. unseen evaluation for a trained model -- reports assignment
# accuracy, trajectory MSE, recovery success rate and the generalization gap G
import sys, os, argparse, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from sim.env import D_STATE, D_OUT, N_UAV, H
from models.nets import StateEncoder, FactorizedConditionEncoder, MonolithicConditionEncoder, DiffusionDenoiser, MLPBaseline
from models.diffusion_utils import DDPMScheduler

SUCCESS_TRAJ_THRESHOLD = 0.05  # normalized-coordinate MSE threshold for "successful" trajectory

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_split(path):
    d = np.load(path, allow_pickle=True)
    return torch.tensor(d["S"]), torch.tensor(d["C"]), torch.tensor(d["Y"]), d["combo_id"]

def decode_and_score(pred, target):
    """pred, target: (B, D_OUT). Split into assignment logits + trajectory, score both."""
    n_assign = N_UAV * (N_UAV + 1)
    assign_pred = pred[:, :n_assign].reshape(-1, N_UAV, N_UAV + 1)
    assign_true = target[:, :n_assign].reshape(-1, N_UAV, N_UAV + 1)
    traj_pred = pred[:, n_assign:]
    traj_true = target[:, n_assign:]

    assign_correct = (assign_pred.argmax(-1) == assign_true.argmax(-1)).float().mean(dim=1)  # per-sample avg over UAVs
    traj_mse = ((traj_pred - traj_true) ** 2).mean(dim=1)

    success = ((assign_correct > 0.99) & (traj_mse < SUCCESS_TRAJ_THRESHOLD)).float()
    return assign_correct.mean().item(), traj_mse.mean().item(), success.mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["A", "B", "C"], required=True)
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat diffusion sampling this many times and average "
                         "(sampling starts from random noise, so B/C vary run to run; "
                         "Model A is deterministic so --runs has no effect on it)")
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(os.path.join("results", f"model_{args.model}.pt"), map_location=device)
    train_args = ckpt["args"]

    S_tr, C_tr, Y_tr, combo_tr = load_split(os.path.join("data", "train.npz"))
    S_te, C_te, Y_te, combo_te = load_split(os.path.join("data", "test.npz"))

    # build the model(s) once, reused across runs
    if args.model == "A":
        model = MLPBaseline(D_STATE, N_UAV, 4, D_OUT).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
    else:
        state_enc = StateEncoder(D_STATE).to(device)
        state_enc.load_state_dict(ckpt["state_enc"])
        if args.model == "C":
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

        assign_acc, traj_mse, success = decode_and_score(pred.cpu(), Y.cpu())
        per_combo = {}
        for c in np.unique(combo):
            idx = torch.tensor(combo == c)
            a, t_, s_ = decode_and_score(pred.cpu()[idx], Y.cpu()[idx])
            per_combo[c] = {"assign_acc": a, "traj_mse": t_, "success_rate": s_}
        return dict(assign_acc=assign_acc, traj_mse=traj_mse, success_rate=success, per_combo=per_combo)

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

        # average per-combo success across runs too
        per_combo_avg = {}
        for c in run_outs[0]["per_combo"]:
            per_combo_avg[c] = {
                "assign_acc": float(np.mean([r["per_combo"][c]["assign_acc"] for r in run_outs])),
                "traj_mse": float(np.mean([r["per_combo"][c]["traj_mse"] for r in run_outs])),
                "success_rate": float(np.mean([r["per_combo"][c]["success_rate"] for r in run_outs])),
            }

        results[split_name] = {
            "assign_acc": mean_acc, "assign_acc_std": std_acc,
            "traj_mse": mean_mse, "traj_mse_std": std_mse,
            "success_rate": mean_succ, "success_rate_std": std_succ,
            "per_combo": per_combo_avg,
        }
        if n_runs > 1:
            print(f"[Model {args.model}] {split_name:6s} -> assign_acc={mean_acc:.3f}+-{std_acc:.3f}  "
                  f"traj_mse={mean_mse:.4f}+-{std_mse:.4f}  success={mean_succ:.3f}+-{std_succ:.3f}  ({n_runs} runs)")
        else:
            print(f"[Model {args.model}] {split_name:6s} -> assign_acc={mean_acc:.3f}  traj_mse={mean_mse:.4f}  success={mean_succ:.3f}")

    results["generalization_gap"] = results["seen"]["success_rate"] - results["unseen"]["success_rate"]
    print(f"[Model {args.model}] Generalization gap G = {results['generalization_gap']:.3f}")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"eval_{args.model}.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
