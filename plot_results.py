"""
plot_results.py
Reads results/eval_A.json, eval_B.json, eval_C.json (run evaluate.py for each first)
and produces the core paper graphs into plots/.

Usage:
    python plot_results.py
"""
import json, os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("plots", exist_ok=True)
MODELS = ["A", "B", "C"]
LABELS = {"A": "MLP (non-diffusion)", "B": "Standard Diffusion (monolithic)", "C": "Factorized Diffusion (ours)"}
COLORS = {"A": "#888888", "B": "#4C72B0", "C": "#DD8452"}

results = {}
for m in MODELS:
    path = os.path.join("results", f"eval_{m}.json")
    if os.path.exists(path):
        with open(path) as f:
            results[m] = json.load(f)

if not results:
    raise SystemExit("No eval_*.json found -- run evaluate.py for each model first.")

# ---------- Graph 1: Seen vs Unseen success rate (grouped bar) ----------
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(results))
width = 0.35
seen_vals = [results[m]["seen"]["success_rate"] for m in results]
unseen_vals = [results[m]["unseen"]["success_rate"] for m in results]
ax.bar(x - width/2, seen_vals, width, label="Seen", color="#55A868")
ax.bar(x + width/2, unseen_vals, width, label="Unseen", color="#C44E52")
ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in results], rotation=15, ha="right")
ax.set_ylabel("Recovery Success Rate")
ax.set_title("Seen vs. Unseen Performance")
ax.legend()
plt.tight_layout()
plt.savefig("plots/01_seen_vs_unseen.png", dpi=150)
plt.close()

# ---------- Graph 2: Generalization Gap comparison ----------
fig, ax = plt.subplots(figsize=(5, 4))
gaps = [results[m]["generalization_gap"] for m in results]
ax.bar([LABELS[m] for m in results], gaps, color=[COLORS[m] for m in results])
ax.set_ylabel("Generalization Gap  G = Seen − Unseen")
ax.set_title("Generalization Gap by Model")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("plots/02_generalization_gap.png", dpi=150)
plt.close()

# ---------- Graph 3: Success rate vs number of failure factors ----------
fig, ax = plt.subplots(figsize=(6, 4))
for m in results:
    per_combo = {}
    per_combo.update(results[m]["seen"]["per_combo"])
    per_combo.update(results[m]["unseen"]["per_combo"])
    by_k = {}
    for combo_str, stats in per_combo.items():
        k = len(combo_str.split("+")) if combo_str != "none" else 0
        by_k.setdefault(k, []).append(stats["success_rate"])
    ks = sorted(by_k.keys())
    means = [np.mean(by_k[k]) for k in ks]
    ax.plot(ks, means, marker="o", label=LABELS[m], color=COLORS[m])
ax.set_xlabel("Number of Simultaneous Failure Factors")
ax.set_ylabel("Recovery Success Rate")
ax.set_title("Success Rate vs. Failure Combination Size")
ax.legend()
plt.tight_layout()
plt.savefig("plots/03_success_vs_num_factors.png", dpi=150)
plt.close()

# ---------- Graph 4: Training loss curves ----------
fig, ax = plt.subplots(figsize=(6, 4))
for m in MODELS:
    hpath = os.path.join("results", f"history_{m}.json")
    if os.path.exists(hpath):
        with open(hpath) as f:
            hist = json.load(f)
        ax.plot(hist, label=LABELS[m], color=COLORS[m])
ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
ax.set_title("Training Loss Curves")
ax.legend()
plt.tight_layout()
plt.savefig("plots/04_training_loss.png", dpi=150)
plt.close()

print("Saved plots to plots/: 01_seen_vs_unseen.png, 02_generalization_gap.png,")
print("                       03_success_vs_num_factors.png, 04_training_loss.png")
