"""
data/generate_dataset.py
Builds the seen/unseen benchmark split:
  TRAIN:      individual failures + pairwise combinations
  VAL_UNSEEN: 2 held-out triples, for hyperparameter tuning only
  TEST_UNSEEN: remaining 2 triples + the quadruple, untouched until final
               reporting (never used for model/hyperparameter selection)
Saves .npz files with S (state), C (failure vector), Y (target).
"""
import sys, os, argparse, itertools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.env import sample_scenario, greedy_baseline_solver, state_vector, target_vector, FACTOR_NAMES

N_EPISODES_PER_COMBO_TRAIN = 1000   # real-MTP scale
N_EPISODES_PER_COMBO_VAL = 300
N_EPISODES_PER_COMBO_TEST = 300

# Fixed, explicit split of the 5 unseen (3- and 4-way) combinations.
# val_unseen is for hyperparameter tuning only; test_unseen must stay untouched
# until final reporting to avoid hyperparameter leakage.
VAL_UNSEEN_COMBOS = [("battery", "comm", "wind"), ("battery", "comm", "nav")]
TEST_UNSEEN_COMBOS = [("battery", "wind", "nav"), ("comm", "wind", "nav"),
                       ("battery", "comm", "wind", "nav")]

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

def report(name, combos, S, C, Y, combo_id):
    print(f"\n{name}: {S.shape[0]} episodes total, S{S.shape} C{C.shape} Y{Y.shape}")
    for c in combos:
        cid = "+".join(c) if c else "none"
        n = int((combo_id == cid).sum())
        print(f"  {cid:35s} {n}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true",
                     help="generate a small dataset (50 episodes/combo) to sanity-check the split")
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__))

    train_combos = all_combos([1, 2])   # singles + pairs -> 4 + 6 = 10 combos
    val_combos = VAL_UNSEEN_COMBOS      # 2 triples
    test_combos = TEST_UNSEEN_COMBOS    # 2 triples + 1 quadruple

    assert set(val_combos).isdisjoint(test_combos)
    assert set(val_combos) | set(test_combos) == set(all_combos([3, 4]))

    print("Train combos      :", train_combos)
    print("Val_unseen combos :", val_combos)
    print("Test_unseen combos:", test_combos)

    if args.tiny:
        n_train = n_val = n_test = 50
    else:
        n_train, n_val, n_test = N_EPISODES_PER_COMBO_TRAIN, N_EPISODES_PER_COMBO_VAL, N_EPISODES_PER_COMBO_TEST

    S_tr, C_tr, Y_tr, id_tr = build_split(train_combos, n_train, seed=42)
    S_va, C_va, Y_va, id_va = build_split(val_combos, n_val, seed=123)
    S_te, C_te, Y_te, id_te = build_split(test_combos, n_test, seed=999)

    np.savez(os.path.join(out_dir, "train.npz"), S=S_tr, C=C_tr, Y=Y_tr, combo_id=id_tr)
    np.savez(os.path.join(out_dir, "val_unseen.npz"), S=S_va, C=C_va, Y=Y_va, combo_id=id_va)
    np.savez(os.path.join(out_dir, "test_unseen.npz"), S=S_te, C=C_te, Y=Y_te, combo_id=id_te)

    report("train", train_combos, S_tr, C_tr, Y_tr, id_tr)
    report("val_unseen", val_combos, S_va, C_va, Y_va, id_va)
    report("test_unseen", test_combos, S_te, C_te, Y_te, id_te)

    print(f"\nTrain set     : {S_tr.shape[0]} episodes -> data/train.npz")
    print(f"Val_unseen set: {S_va.shape[0]} episodes -> data/val_unseen.npz")
    print(f"Test_unseen set: {S_te.shape[0]} episodes -> data/test_unseen.npz")
