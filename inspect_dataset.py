"""
inspect_dataset.py
Quick readable preview of train.npz / test.npz -- since these are binary
NumPy files, VS Code can't preview them directly. Run this instead:

    python inspect_dataset.py data/train.npz
    python inspect_dataset.py data/test.npz
"""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "data/train.npz"
d = np.load(path, allow_pickle=True)

print(f"\n=== {path} ===")
print("Keys stored:", d.files)
print(f"Total episodes: {d['S'].shape[0]}")
print(f"S (state) shape       : {d['S'].shape}   -> (episodes, N_uav, d_state)")
print(f"C (failure cond) shape: {d['C'].shape}   -> (episodes, 4 factors: battery,comm,wind,nav)")
print(f"Y (target) shape      : {d['Y'].shape}   -> (episodes, flattened assignment+trajectory)")

print("\nCombination breakdown:")
combos, counts = np.unique(d["combo_id"], return_counts=True)
for c, n in zip(combos, counts):
    print(f"  {c:25s} : {n} episodes")

print("\n--- Example: episode 0 ---")
print("combo_id:", d["combo_id"][0])
print("failure vector [battery, comm, wind, nav]:", d["C"][0])
print("UAV states (position_x, position_y, battery, comm, task_onehot x6):")
for i, row in enumerate(d["S"][0]):
    print(f"  UAV {i}: {row}")
