"""
sim/env.py
Lightweight 2D kinematic multi-UAV scenario generator + baseline solver.
No physics engine needed — pure NumPy, runs instantly on CPU.

A "scenario" = N UAVs, each with a start position and a nominal assigned task
(a target position), plus a failure-condition vector describing which of the
4 failure factors are active and how severe.

The baseline solver computes a REFERENCE recovery plan:
  - task reassignment (if a UAV is too failed to complete its task, its task
    is greedily handed to the nearest surviving UAV)  <-- cross-agent propagation
  - a straight-line (perturbed) trajectory for every UAV toward its
    (possibly reassigned) task, perturbed by wind/nav/comm effects.

This reference plan is the supervision label (Y_0) the diffusion model is
trained to imitate and later must generalize with, for unseen failure
combinations.
"""

import numpy as np

N_UAV = 6          # number of UAVs (== number of tasks, 1:1 nominal assignment)
H = 10             # trajectory horizon (timesteps)
WORLD_SIZE = 100.0 # arena is [0, WORLD_SIZE] x [0, WORLD_SIZE]

FACTOR_NAMES = ["battery", "comm", "wind", "nav"]

# severity threshold above which a UAV is considered "too failed" to
# complete its own task and must hand it off to a neighbour
REASSIGN_THRESHOLD = 0.55


def sample_scenario(rng, active_factors, severity_range=(0.5, 1.0)):
    """
    active_factors: subset of FACTOR_NAMES, e.g. ["battery"], ["battery","wind"], etc.
    Returns a dict describing one scenario.
    """
    start_pos = rng.uniform(10, WORLD_SIZE - 10, size=(N_UAV, 2))
    task_pos = rng.uniform(10, WORLD_SIZE - 10, size=(N_UAV, 2))  # task i nominally for UAV i
    battery = rng.uniform(0.7, 1.0, size=N_UAV)
    comm_ok = np.ones(N_UAV)

    # failure severities: 0 if inactive, sampled in severity_range if active
    sev = {f: 0.0 for f in FACTOR_NAMES}
    for f in active_factors:
        sev[f] = rng.uniform(*severity_range)

    # which UAV(s) are directly affected by an agent-level failure
    # (battery / nav are per-agent; comm affects a random agent too;
    #  wind is a GLOBAL disturbance affecting everyone)
    affected_uav = int(rng.integers(0, N_UAV))

    if sev["battery"] > 0:
        battery[affected_uav] = max(0.0, battery[affected_uav] - sev["battery"] * 0.9)
    if sev["comm"] > 0:
        comm_ok[affected_uav] = 0.0
    if sev["nav"] > 0:
        start_pos[affected_uav] += rng.normal(0, sev["nav"] * 8, size=2)  # position bias/drift

    failure_vec = np.array([sev[f] for f in FACTOR_NAMES], dtype=np.float32)  # [battery,comm,wind,nav]

    return dict(
        start_pos=start_pos.astype(np.float32),
        task_pos=task_pos.astype(np.float32),
        battery=battery.astype(np.float32),
        comm_ok=comm_ok.astype(np.float32),
        failure_vec=failure_vec,
        affected_uav=affected_uav,
    )


def greedy_baseline_solver(scenario, rng):
    """
    Reference/optimal recovery plan generator (NOT learned — classical greedy logic).
    Produces:
      assignment: (N,) int array, assignment[i] = index of task assigned to UAV i
      traj:       (N, H, 2) float array, waypoints per UAV
    """
    start_pos = scenario["start_pos"]
    task_pos = scenario["task_pos"].copy()
    battery = scenario["battery"]
    sev_battery, sev_comm, sev_wind, sev_nav = scenario["failure_vec"]
    affected = scenario["affected_uav"]

    assignment = np.arange(N_UAV)  # nominal: UAV i -> task i

    # --- cross-agent task reassignment ---
    # A UAV is "too failed" if its own battery is critically low AND
    # (compounded by comm loss, since it can't coordinate a handoff cleanly).
    # This nonlinear interaction is exactly what a FACTORIZED model has to
    # learn to compose correctly from separately-seen factors.
    # --- cross-agent task reassignment ---
    # A UAV is "too failed" to keep its task if EITHER:
    #   (a) its battery is critically low, or
    #   (b) it has lost communication badly enough that it cannot be
    #       coordinated for a handoff at all.
    # Wind/nav stay as trajectory-level disturbances only (they perturb the
    # path, they don't by themselves justify pulling a UAV off its task).
    # This ensures every test combo (including comm+wind+nav, which has no
    # battery factor) still exercises the cross-agent reassignment behaviour.
    COMM_CRITICAL_SEVERITY = 0.6
    battery_critical = sev_battery > 0 and battery[affected] < (1 - REASSIGN_THRESHOLD)
    comm_critical = sev_comm > COMM_CRITICAL_SEVERITY
    is_critical = battery_critical or comm_critical

    if is_critical:
        failed_uav = affected
        failed_task = assignment[failed_uav]
        # find nearest surviving UAV (by current position) to take over the task
        dists = np.linalg.norm(start_pos - start_pos[failed_uav], axis=1)
        dists[failed_uav] = np.inf
        helper = int(np.argmin(dists))
        # helper now must do BOTH tasks in sequence: its own, then the failed one
        # (for simplicity in this v1: helper's trajectory targets midpoint-then-handoff task)
        # We encode this by making helper's final target = failed_task position,
        # and the failed UAV just returns to base (0,0)-ish / holds position.
        assignment[helper] = failed_task  # helper re-targets to the failed UAV's task
        assignment[failed_uav] = -1       # -1 = "return to base / hold", not doing a task

    # --- build trajectories ---
    traj = np.zeros((N_UAV, H, 2), dtype=np.float32)
    for i in range(N_UAV):
        p0 = start_pos[i].copy()
        if assignment[i] == -1:
            target = np.array([WORLD_SIZE / 2, WORLD_SIZE / 2])  # "return to base"
        else:
            target = task_pos[assignment[i]]

        for h in range(H):
            frac = (h + 1) / H
            pt = p0 * (1 - frac) + target * frac

            # wind: global lateral drift, grows with severity, same-ish direction for all
            if sev_wind > 0:
                wind_dir = np.array([1.0, 0.3])  # fixed wind direction for reproducibility
                pt += wind_dir * sev_wind * 3.0 * frac

            # comm loss on this agent -> jerkier/noisier path (can't get clean corrections)
            if i == affected and sev_comm > 0:
                pt += rng.normal(0, sev_comm * 1.5, size=2)

            traj[i, h] = pt

    return assignment, traj


def state_vector(scenario):
    """
    Flatten scenario into the model's input state tensor S_t: (N, d_state)
    d_state = 2 (pos) + 1 (battery) + 1 (comm) + 6 (nominal task one-hot) = 10
    """
    N = N_UAV
    onehot = np.eye(N, dtype=np.float32)  # nominal task i -> UAV i
    S = np.concatenate([
        scenario["start_pos"] / WORLD_SIZE,      # normalize to [0,1]
        scenario["battery"][:, None],
        scenario["comm_ok"][:, None],
        onehot,
    ], axis=1)
    return S.astype(np.float32)  # (N, 10)


def target_vector(assignment, traj):
    """
    Flatten (assignment, traj) into the model's output target Y_0.
    assignment (N,) with -1 allowed -> one-hot over (N+1) classes (last = "return to base")
    traj (N,H,2) normalized by WORLD_SIZE
    Returns flat vector of size N*(N+1) + N*H*2
    """
    N = N_UAV
    A_onehot = np.zeros((N, N + 1), dtype=np.float32)
    for i, a in enumerate(assignment):
        A_onehot[i, a if a >= 0 else N] = 1.0
    T_norm = (traj / WORLD_SIZE).astype(np.float32)
    return np.concatenate([A_onehot.flatten(), T_norm.flatten()]).astype(np.float32)


D_STATE = 10
D_OUT = N_UAV * (N_UAV + 1) + N_UAV * H * 2  # assignment logits + trajectory
