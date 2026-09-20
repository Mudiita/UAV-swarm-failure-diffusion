# Hyperparameter sweep for Model C's dropout_p, selecting ONLY on val_unseen.
# test_unseen is never touched here -- it stays untouched for final reporting.
#
# Each (dropout_p, seed) run passes --out_tag "dp{dp}_seed{seed}" to both
# train.py and evaluate.py, so the checkpoint/history/eval files are written
# directly at their final tagged path -- there's no intermediate untagged
# file and therefore no window where two different dropout_p values could
# ever contend for the same path.
import os, sys, argparse, subprocess, json
import numpy as np

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dropout_ps", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--epochs", type=int, required=True)
    args = ap.parse_args()

    python = sys.executable
    summary = {}

    for dp in args.dropout_ps:
        seed_evals = []
        for seed in args.seeds:
            tag = f"dp{dp}_seed{seed}"
            tagged_ckpt = os.path.join("results", f"model_C_{tag}.pt")
            tagged_eval = os.path.join("results", f"eval_C_{tag}_val_unseen.json")

            if os.path.exists(tagged_eval):
                print(f"-- skipping {tag}, already done ({tagged_eval} exists) --")
            else:
                run([python, "train.py", "--model", "C", "--dropout_p", str(dp),
                     "--seed", str(seed), "--epochs", str(args.epochs), "--out_tag", tag])
                run([python, "evaluate.py", "--model", "C", "--checkpoint", tagged_ckpt,
                     "--split", "val_unseen", "--out_tag", tag])

            with open(tagged_eval) as f:
                seed_evals.append(json.load(f))

        unseen_success = [r["unseen"]["success_rate"] for r in seed_evals]
        gaps = [r["generalization_gap"] for r in seed_evals]
        summary[str(dp)] = {
            "unseen_success_mean": float(np.mean(unseen_success)),
            "unseen_success_std": float(np.std(unseen_success)),
            "gap_mean": float(np.mean(gaps)),
            "gap_std": float(np.std(gaps)),
        }

    print("\n=== dropout_p sweep summary (val_unseen only, test_unseen untouched) ===")
    for dp_str, s in summary.items():
        print(f"dropout_p={dp_str}  unseen_success={s['unseen_success_mean']:.3f}+-{s['unseen_success_std']:.3f}  "
              f"gap={s['gap_mean']:.3f}+-{s['gap_std']:.3f}")

    best_dp = max(summary, key=lambda k: summary[k]["unseen_success_mean"])
    print(f"\nBest dropout_p by val_unseen success: {best_dp}")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "dropout_sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved -> results/dropout_sweep_summary.json")

if __name__ == "__main__":
    main()
