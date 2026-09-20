# Per-combination success_rate table across models A/B/C/D for a given split,
# built from existing eval_*.json files (evaluate.py's per_combo breakdown).
# When multiple seeds' eval files exist for a model, their per-combo
# success_rate is averaged. Hyperparameter-sweep artifacts
# (eval_{model}_dp*_seed*_{split}.json) are intentionally excluded by the glob
# below -- this reports finalized per-seed model evals, not sweep runs.
import os, argparse, glob, json, csv
import numpy as np

MODELS = ["A", "B", "C", "D"]

def load_model_evals(model, split):
    pattern = os.path.join("results", f"eval_{model}_seed*_{split}.json")
    paths = sorted(glob.glob(pattern))
    evals = []
    for p in paths:
        with open(p) as f:
            evals.append(json.load(f))
    return paths, evals

def combo_sort_key(combo):
    return (len(combo.split("+")), combo)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val_unseen", "test_unseen"], required=True)
    args = ap.parse_args()

    model_data = {}  # model -> {combo: [success_rate, ...across seeds]}
    combos = set()
    for model in MODELS:
        paths, evals = load_model_evals(model, args.split)
        if not evals:
            continue
        combo_vals = {}
        for r in evals:
            for combo, stats in r["unseen"]["per_combo"].items():
                combo_vals.setdefault(combo, []).append(stats["success_rate"])
                combos.add(combo)
        model_data[model] = combo_vals
        print(f"[report_per_combo] model {model}: {len(paths)} eval file(s) -> {[os.path.basename(p) for p in paths]}")

    if not model_data:
        print(f"No eval_{{model}}_seed*_{args.split}.json files found for any model. Nothing to report.")
        return

    models_present = [m for m in MODELS if m in model_data]
    sorted_combos = sorted(combos, key=combo_sort_key)

    header = f"{'combo':35s}" + "".join(f"{m:>12s}" for m in models_present)
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for combo in sorted_combos:
        row_vals = []
        for m in models_present:
            vals = model_data[m].get(combo)
            row_vals.append(float(np.mean(vals)) if vals else None)
        cells = "".join(f"{(f'{v:.3f}' if v is not None else 'NA'):>12s}" for v in row_vals)
        print(f"{combo:35s}{cells}")
        rows.append([combo] + [("" if v is None else f"{v:.4f}") for v in row_vals])

    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", f"per_combo_success_{args.split}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combo"] + models_present)
        w.writerows(rows)
    print(f"\nSaved -> {out_csv}")

if __name__ == "__main__":
    main()
