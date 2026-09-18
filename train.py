# trains one of the three planners: A = MLP baseline, B = monolithic diffusion,
# C = factorized diffusion (ours). run separately per model, e.g.
#   python train.py --model C --epochs 300 --dropout_p 0.2
import sys, os, argparse, json
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from sim.env import D_STATE, D_OUT, N_UAV
from models.nets import StateEncoder, FactorizedConditionEncoder, MonolithicConditionEncoder, DiffusionDenoiser, MLPBaseline
from models.diffusion_utils import DDPMScheduler

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_split(path):
    d = np.load(path, allow_pickle=True)
    return torch.tensor(d["S"]), torch.tensor(d["C"]), torch.tensor(d["Y"]), d["combo_id"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["A", "B", "C"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dropout_p", type=float, default=0.2)   # factor-dropout prob (Model C only)
    ap.add_argument("--T", type=int, default=100)              # diffusion timesteps (B, C)
    args = ap.parse_args()

    device = get_device()
    print("Device:", device)

    S, C, Y, _ = load_split(os.path.join("data", "train.npz"))
    S, C, Y = S.to(device), C.to(device), Y.to(device)
    n = S.shape[0]

    state_enc = StateEncoder(D_STATE).to(device)

    if args.model == "A":
        model = MLPBaseline(D_STATE, N_UAV, 4, D_OUT).to(device)
        params = list(model.parameters())
    else:
        if args.model == "C":
            cond_enc = FactorizedConditionEncoder(dropout_p=args.dropout_p).to(device)
        else:  # B
            cond_enc = MonolithicConditionEncoder().to(device)
        denoiser = DiffusionDenoiser(D_OUT, 64, cond_enc.out_dim).to(device)
        scheduler = DDPMScheduler(T=args.T, device=device)
        params = list(state_enc.parameters()) + list(cond_enc.parameters()) + list(denoiser.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i+args.batch_size]
            s_b, c_b, y_b = S[idx], C[idx], Y[idx]

            if args.model == "A":
                pred = model(s_b, c_b)
                loss = loss_fn(pred, y_b)
            else:
                state_ctx = state_enc(s_b)
                cond_ctx = cond_enc(c_b, training=True)
                B_ = y_b.shape[0]
                t = torch.randint(0, args.T, (B_,), device=device)
                noise = torch.randn_like(y_b)
                y_t = scheduler.add_noise(y_b, t, noise)
                eps_pred = denoiser(y_t, t, state_ctx, cond_ctx)
                loss = loss_fn(eps_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        history.append(avg_loss)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[Model {args.model}] epoch {epoch:3d}  loss={avg_loss:.5f}")

    os.makedirs("results", exist_ok=True)
    ckpt = {"args": vars(args)}
    if args.model == "A":
        ckpt["model_state"] = model.state_dict()
    else:
        ckpt["state_enc"] = state_enc.state_dict()
        ckpt["cond_enc"] = cond_enc.state_dict()
        ckpt["denoiser"] = denoiser.state_dict()
    torch.save(ckpt, os.path.join("results", f"model_{args.model}.pt"))
    with open(os.path.join("results", f"history_{args.model}.json"), "w") as f:
        json.dump(history, f)
    print(f"Saved -> results/model_{args.model}.pt")

if __name__ == "__main__":
    main()
