# Empirically tests whether Model C's val_unseen success_rate and
# generalization gap depend on training-set size. Only train.npz is
# resampled at each size; val_unseen.npz/test_unseen.npz are NEVER
# regenerated or touched, so the held-out episodes are identical across all
# sizes and the comparison isn't confounded by different held-out data.
#
# train.npz is restored to whatever it was before this script ran once the
# experiment finishes (success or failure), so the canonical dataset used by
# every other script is left untouched afterward.
#
# Each (size, seed) run passes --out_tag "size{size}_seed{seed}" to both
# train.py and evaluate.py, so checkpoint/history/eval files are written
# directly at their final tagged path -- no intermediate untagged file, no
# window where two different sizes could contend for the same path.
import os, sys, argparse, subprocess, shutil, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data.generate_dataset import all_combos, build_split

TRAIN_COMBOS = all_combos([1, 2])
TRAIN_PATH = os.path.join("data", "train.npz")

def generate_train_at_size(n_per_combo, seed):
    S, C, Y, combo_id = build_split(TRAIN_COMBOS, n_per_combo, seed=seed)
    np.savez(TRAIN_PATH, S=S, C=C, Y=Y, combo_id=combo_id)
    return S.shape[0]

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", required=True,
                    help="episodes per combo for train.npz, e.g. 500 1000 2000")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--dropout_p", type=float, required=True,
                    help="best-known dropout_p for Model C, from the dropout_p sweep")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--data_seed", type=int, default=42,
                    help="seed for resampling train.npz at each size (independent of model training seed)")
    args = ap.parse_args()

    python = sys.executable
    summary = {}

    backup_path = None
    if os.path.exists(TRAIN_PATH):
        backup_path = TRAIN_PATH + ".bak_before_size_experiment"
        shutil.copy(TRAIN_PATH, backup_path)
        print(f"Backed up current train.npz -> {backup_path}")

    try:
        for size in args.sizes:
            n_episodes = generate_train_at_size(size, seed=args.data_seed)
            print(f"\n=== train size = {size}/combo ({n_episodes} episodes total) ===")

            seed_evals = []
            for seed in args.seeds:
                tag = f"size{size}_seed{seed}"
                run([python, "train.py", "--model", "C", "--dropout_p", str(args.dropout_p),
                     "--seed", str(seed), "--epochs", str(args.epochs), "--out_tag", tag])

                tagged_ckpt = os.path.join("results", f"model_C_{tag}.pt")
                run([python, "evaluate.py", "--model", "C", "--checkpoint", tagged_ckpt,
                     "--split", "val_unseen", "--out_tag", tag])

                tagged_eval = os.path.join("results", f"eval_C_{tag}_val_unseen.json")
                with open(tagged_eval) as f:
                    seed_evals.append(json.load(f))

            unseen_success = [r["unseen"]["success_rate"] for r in seed_evals]
            gaps = [r["generalization_gap"] for r in seed_evals]
            summary[size] = {
                "unseen_success_mean": float(np.mean(unseen_success)),
                "unseen_success_std": float(np.std(unseen_success)),
                "gap_mean": float(np.mean(gaps)),
                "gap_std": float(np.std(gaps)),
            }
    finally:
        if backup_path and os.path.exists(backup_path):
            shutil.move(backup_path, TRAIN_PATH)
            print(f"\nRestored original train.npz from backup")

    print(f"\n=== dataset-size experiment summary (val_unseen only; Model C, dropout_p={args.dropout_p}) ===")
    for size, s in summary.items():
        print(f"size={size:5d}/combo  unseen_success={s['unseen_success_mean']:.3f}+-{s['unseen_success_std']:.3f}  "
              f"gap={s['gap_mean']:.3f}+-{s['gap_std']:.3f}")

    sizes_sorted = sorted(summary)
    print()
    for a, b in zip(sizes_sorted, sizes_sorted[1:]):
        delta = abs(summary[b]["gap_mean"] - summary[a]["gap_mean"])
        note = "< 0.02: diminishing returns" if delta < 0.02 else ">= 0.02: still meaningfully changing"
        print(f"gap change {a} -> {b}: {delta:.4f} ({note})")
        if a == 1000 and b == 2000:
            verdict = "YES, gap change < 0.02" if delta < 0.02 else "NO, gap change >= 0.02"
            print(f"[EXPLICIT CHECK] Does 1000->2000 change the gap by less than 0.02? {verdict} (delta={delta:.4f})")

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "dataset_size_experiment_summary.json"), "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    print("Saved -> results/dataset_size_experiment_summary.json")

if __name__ == "__main__":
    main()
