# aggregates eval_{model}_seed{seed}_{split}.json files across seeds and
# reports mean +/- std of seen_success, unseen_success and generalization_gap.
# usage: python aggregate_seeds.py --model C --seeds 0 1 2 --split test_unseen
import os, argparse, json
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--split", choices=["train", "val_unseen", "test_unseen"], default="test_unseen")
    args = ap.parse_args()

    seen_success, unseen_success, gaps = [], [], []
    missing = []
    for seed in args.seeds:
        path = os.path.join("results", f"eval_{args.model}_seed{seed}_{args.split}.json")
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as f:
            r = json.load(f)
        seen_success.append(r["seen"]["success_rate"])
        unseen_success.append(r["unseen"]["success_rate"])
        gaps.append(r["generalization_gap"])

    if missing:
        raise FileNotFoundError(f"Missing eval files: {missing}")

    def fmt(vals):
        return f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}"

    print(f"[Model {args.model}] split={args.split}  n_seeds={len(args.seeds)} {args.seeds}")
    print(f"  seen_success       = {fmt(seen_success)}")
    print(f"  unseen_success     = {fmt(unseen_success)}")
    print(f"  generalization_gap = {fmt(gaps)}")

if __name__ == "__main__":
    main()
