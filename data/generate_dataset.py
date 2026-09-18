"""
data/generate_dataset.py
Builds the seen/unseen benchmark split:
  TRAIN: individual failures + pairwise combinations
  TEST:  unseen triple + quadruple combinations
Saves .npz files with S (state), C (failure vector), Y (target).
"""
import sys, os, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.env import sample_scenario, greedy_baseline_solver, state_vector, target_vector, FACTOR_NAMES

N_EPISODES_PER_COMBO_TRAIN = 1000   # real-MTP scale (was 300 for the quick prototype)
N_EPISODES_PER_COMBO_TEST = 300     # real-MTP scale (was 100)

def all_combos(k_list):
    combos = []
    for k in k_list:
        combos.extend(itertools.combinations(FACTOR_NAMES, k))
    return combos

def build_split(combos, n_per_combo, seed):
    rng = np.random.default_rng(seed)
    S_list, C_list, Y_list, combo_id_list = [], [], [], []
    for combo in combos:
        for _ in range(n_per_combo):
            scenario = sample_scenario(rng, active_factors=list(combo))
            assignment, traj = greedy_baseline_solver(scenario, rng)
            S_list.append(state_vector(scenario))
            C_list.append(scenario["failure_vec"])
            Y_list.append(target_vector(assignment, traj))
            combo_id_list.append("+".join(combo) if combo else "none")
    return (np.stack(S_list), np.stack(C_list), np.stack(Y_list), np.array(combo_id_list))

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__))

    train_combos = all_combos([1, 2])   # singles + pairs  -> 4 + 6 = 10 combos
    test_combos = all_combos([3, 4])    # triples + quad    -> 4 + 1 = 5 combos

    print("Train combos:", train_combos)
    print("Test combos :", test_combos)

    S_tr, C_tr, Y_tr, id_tr = build_split(train_combos, N_EPISODES_PER_COMBO_TRAIN, seed=42)
    S_te, C_te, Y_te, id_te = build_split(test_combos, N_EPISODES_PER_COMBO_TEST, seed=999)

    np.savez(os.path.join(out_dir, "train.npz"), S=S_tr, C=C_tr, Y=Y_tr, combo_id=id_tr)
    np.savez(os.path.join(out_dir, "test.npz"), S=S_te, C=C_te, Y=Y_te, combo_id=id_te)

    print(f"Train set: {S_tr.shape[0]} episodes -> data/train.npz")
    print(f"Test set : {S_te.shape[0]} episodes -> data/test.npz")
