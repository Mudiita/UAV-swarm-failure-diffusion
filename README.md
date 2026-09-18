# UAV Compositional-Generalization Diffusion — Starter Pipeline

This is a **working, minimal skeleton** that runs end-to-end on a MacBook Air M3
(CPU/MPS, no GPU cluster needed). It's a *prototype* to prove the pipeline works —
you will need to scale up (more data, more epochs, bigger model) for real MTP-quality
results. Every number you see out of the box is a placeholder from ~150 epochs of
training on tiny synthetic data.

## Setup (one time)
```bash
pip install -r requirements.txt
```
(If not using a virtual environment, you may need `pip install -r requirements.txt --break-system-packages` instead.)

## 1. Generate the dataset
```bash
python data/generate_dataset.py
```
This creates `data/train.npz` (singles + pairs of failures) and `data/test.npz`
(unseen triples + quadruple). Currently set to 1000 episodes/combo for train
(10,000 total) and 300/combo for test (1,500 total) -- real-MTP scale, not the
quick-prototype scale. Adjust `N_EPISODES_PER_COMBO_TRAIN` /
`_TEST` at the top of the script if you want to go bigger or smaller.

## 2. Train each model
```bash
python train.py --model A --epochs 150     # MLP baseline, no diffusion
python train.py --model B --epochs 150     # standard (monolithic) diffusion
python train.py --model C --epochs 150     # factorized diffusion -- YOUR contribution
```
Useful flags to change for experiments/ablations:
- `--dropout_p 0.1` / `0.3` — factor-dropout probability (Model C only) — this is
  the knob that controls HOW MUCH the model is forced to learn factors independently.
- `--T 50` / `200` — number of diffusion timesteps.
- `--lr`, `--batch_size`, `--epochs` — standard training knobs.

Each run saves `results/model_X.pt` (weights) and `results/history_X.json`
(loss curve, used for Graph 4).

## 3. Evaluate each model (seen vs. unseen)
```bash
python evaluate.py --model A
python evaluate.py --model B
python evaluate.py --model C
```
Prints assignment-accuracy, trajectory-MSE, success-rate, and the
**generalization gap** for both seen (train combos) and unseen (test combos).
Saves `results/eval_X.json`.

Diffusion sampling (Models B/C) starts from random noise, so results vary
slightly run to run. Use `--runs 5` to average over 5 samples and report
mean +/- std -- more reliable for reporting in the paper than a single run:
```bash
python evaluate.py --model C --runs 5
```

## 4. Generate all graphs
```bash
python plot_results.py
```
Produces in `plots/`:
- `01_seen_vs_unseen.png` — grouped bar, seen vs unseen success per model
- `02_generalization_gap.png` — bar chart of G per model (your main RQ2 evidence)
- `03_success_vs_num_factors.png` — line graph, success rate vs. # simultaneous failures
- `04_training_loss.png` — training curves for all 3 models

## What to compare / how to read results

| Comparison | What it tells you |
|---|---|
| Model A vs B/C on **unseen** | Does diffusion help at all vs. a plain regressor? |
| Model B vs C on **unseen** (this is the KEY one) | Does **factorization** improve generalization (RQ2)? |
| `generalization_gap` (Model B) vs (Model C) | Smaller gap for C = your novelty is supported |
| Graph 3, C's curve vs B's curve as factor-count increases | Does the factorized model degrade more slowly as failures compound? |

## Next steps to get real (paper-quality) numbers
1. Increase `N_EPISODES_PER_COMBO_TRAIN` to 1000+, `_TEST` to 300+.
2. Increase `--epochs` to 300-1000 (watch the loss curve in Graph 4 — it should
   still be decreasing, not flat, when you stop).
3. If training is too slow on CPU, move `train.py` to Google Colab (free GPU) —
   just re-upload `data/*.npz` and run the same commands.
4. Once Model C clearly beats Model B on unseen combos with a smaller
   generalization gap, THAT result is your core paper finding (RQ1 + RQ2).
5. Single-shot distillation (Model D / RQ3) is intentionally NOT included here —
   it is a secondary extension. Once Model C is solid, come back for the
   distillation script (teacher = trained Model C, student = one-step generator).

## File map
```
sim/env.py              - 2D kinematic scenario generator + greedy baseline solver
data/generate_dataset.py - builds train/test .npz files
models/nets.py          - StateEncoder, Factorized/Monolithic condition encoders,
                           DiffusionDenoiser, MLPBaseline
models/diffusion_utils.py - DDPM noise scheduler (forward + reverse sampling)
train.py                - trains Model A / B / C
evaluate.py             - computes seen/unseen metrics + generalization gap
plot_results.py         - generates all 4 graphs
inspect_dataset.py      - readable preview of a .npz dataset file
requirements.txt        - pinned Python dependencies
.gitignore              - excludes venv/, generated data, checkpoints, plots from git
```
