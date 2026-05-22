# =========================================================
# SCHEDULER  –  D-MODULE
#
# Capacity-Constrained Region Partitioning
# (paper Section "Capacity-constrained task region
#  partitioning", eq 3–8, Fig. 1)
#
# Solves:
#   min  Σ_i Σ_u (α·d(p_u,g_i) − γ·w^pri_i) x_ui
#        + λ_TV Σ_{(i,j)∈E} Σ_u |x_ui − x_uj|
#
#   s.t. Σ_u x_ui = 1          ∀i  (eq 4)
#        Σ_i ω_{k,i} x_ui ≤ C_{u,k}  ∀u,k  (eq 5)
#        x_ui ∈ {0,1}           (eq 6)
#
# Solved via iterative Lagrange-relaxed power-diagram:
#   δ_ui = α·d(p_u,g_i) − γ·w^pri_i + Σ_k μ_{u,k}·ω_{k,i} (eq 7)
#   μ_{u,k} ← [μ_{u,k} + ρ(usage − capacity)]+            (eq 8)
#
# After each iteration local 1-swap improves TV term
# (spatial compactness).
# =========================================================

import math

from config import (
    ALPHA,
    GAMMA,
    RHO,
    LAMBDA_TV,
    ITERATIONS,
    ENERGY_PER_METER,
    UAV_SPEED,
)


# ----------------------------------------------------------
# HELPER: travel energy cost for UAV u to reach task i
# ----------------------------------------------------------

def travel_energy(uav, task):
    """
    Energy consumed flying from uav position to task.
    J = ENERGY_PER_METER × distance
    """
    dist = math.hypot(uav.x - task.x, uav.y - task.y)
    return ENERGY_PER_METER * dist


def travel_time(uav, task):
    """Seconds to fly from uav position to task."""
    dist = math.hypot(uav.x - task.x, uav.y - task.y)
    return dist / UAV_SPEED


# ----------------------------------------------------------
# CAPACITY FEASIBILITY CHECK  (eq 5)
#
# Checks all three dimensions:
#   energy      : Σ ω_E,i + travel_energy ≤ C_u,E
#   hover_time  : Σ ω_H,i + travel_time   ≤ C_u,H
#   compute     : Σ ω_F,i                 ≤ C_u,F
# ----------------------------------------------------------

def is_feasible(uav, task):
    """
    Returns True iff assigning *task* to *uav* keeps all
    three resource dimensions within their capacity limits.
    Includes travel overhead (energy + time to reach task).
    Also enforces flight-range constraint (eq 9) and
    type-compatibility constraint (eq 13).
    """

    # Type-compatibility  ϕ_{u,i} = 1 check  (eq 13 / 19)
    if not uav.is_compatible(task):
        return False

    # Flight range constraint  (eq 9)
    dist = math.hypot(uav.x - task.x, uav.y - task.y)
    if dist > uav.max_flight_range:
        return False

    # Accumulated usage so far
    used_energy  = sum(
        t.energy_cost + ENERGY_PER_METER *
        math.hypot(uav.x - t.x, uav.y - t.y)
        for t in uav.assigned_tasks
    )
    used_hover   = sum(t.hover_time  for t in uav.assigned_tasks)
    used_compute = sum(t.compute_load for t in uav.assigned_tasks)

    # Travel cost to this new task
    t_energy = travel_energy(uav, task)
    t_time   = travel_time(uav, task)

    energy_ok  = (used_energy  + task.energy_cost + t_energy
                  <= uav.max_energy)
    hover_ok   = (used_hover   + task.hover_time  + t_time
                  <= uav.max_hover_time)
    compute_ok = (used_compute + task.compute_load
                  <= uav.max_compute)

    return energy_ok and hover_ok and compute_ok


# ----------------------------------------------------------
# GENERALIZED COST  δ_ui  (eq 7)
#
# δ_ui = α·d(p_u,g_i) − γ·w^pri_i + Σ_k μ_{u,k}·ω_{k,i}
# ----------------------------------------------------------

def generalized_cost(uav, task):
    """
    Full generalized cost including:
      • distance term  α·d(...)
      • priority attraction  −γ·w^pri_i
      • Lagrange penalty  Σ_k μ_{u,k}·ω_{k,i}

    Priority weight w^pri_i:
      priority 1  →  100  (critical)
      priority 2  →   60  (important)
      priority 3  →   20  (routine)
    """

    distance = math.hypot(uav.x - task.x, uav.y - task.y)

    priority_weight_map = {1: 100, 2: 60, 3: 20}
    priority_weight = priority_weight_map.get(task.priority, 20)

    # Lagrange multiplier penalty  Σ_k μ_{u,k} · ω_{k,i}
    lagrange_penalty = (
        uav.mu_energy  * (task.energy_cost + travel_energy(uav, task))
        + uav.mu_hover * (task.hover_time  + travel_time(uav, task))
        + uav.mu_compute * task.compute_load
    )

    return (
        ALPHA * distance
        - GAMMA * priority_weight
        + lagrange_penalty
    )


# ----------------------------------------------------------
# TV REGULARISATION TERM  (eq 3, λ_TV component)
#
# Approximated as distance from proposed task to the
# centroid of the UAV's current region.  Adding a task
# far from the centroid increases the TV penalty.
# ----------------------------------------------------------

def compactness_penalty(uav, task):
    """
    λ_TV · ||task - centroid(region_u)||
    Returns 0 for the first task (no region yet).
    """
    if not uav.assigned_tasks:
        return 0.0

    n = len(uav.assigned_tasks)
    cx = sum(t.x for t in uav.assigned_tasks) / n
    cy = sum(t.y for t in uav.assigned_tasks) / n

    centroid_dist = math.hypot(task.x - cx, task.y - cy)
    return LAMBDA_TV * centroid_dist


# ----------------------------------------------------------
# 1-SWAP LOCAL REFINEMENT  (post-assignment TV improvement)
#
# After all tasks are assigned, try swapping tasks between
# pairs of UAVs if the swap reduces the total TV (centroid-
# distance) objective without violating any capacity.
# ----------------------------------------------------------

def local_swap_refinement(uavs, max_swaps=20):
    """
    Iterates over all pairs (u1, u2) and tries swapping
    each task of u1 with each task of u2.  Accepts the
    swap if it improves total compactness and both UAVs
    remain feasible after the swap.
    """

    def region_tv(uav):
        """Sum of distances from tasks to their UAV centroid."""
        if len(uav.assigned_tasks) < 2:
            return 0.0
        n  = len(uav.assigned_tasks)
        cx = sum(t.x for t in uav.assigned_tasks) / n
        cy = sum(t.y for t in uav.assigned_tasks) / n
        return sum(
            math.hypot(t.x - cx, t.y - cy)
            for t in uav.assigned_tasks
        )

    def swap_feasible(u1, t1, u2, t2):
        """Check capacity feasibility after swap."""
        # Simulate: u1 loses t1 gains t2; u2 loses t2 gains t1
        def total_usage(uav, exclude, include):
            tasks_after = [
                t for t in uav.assigned_tasks if t is not exclude
            ] + [include]
            e = sum(
                t.energy_cost + ENERGY_PER_METER *
                math.hypot(uav.x - t.x, uav.y - t.y)
                for t in tasks_after
            )
            h = sum(t.hover_time   for t in tasks_after)
            f = sum(t.compute_load for t in tasks_after)
            return e, h, f

        def within_cap(uav, e, h, f):
            return (
                e <= uav.max_energy
                and h <= uav.max_hover_time
                and f <= uav.max_compute
                and uav.is_compatible(
                    type('X', (), {'task_type': t2.task_type})()
                )
            )

        e1, h1, f1 = total_usage(u1, t1, t2)
        e2, h2, f2 = total_usage(u2, t2, t1)

        cap1 = (
            e1 <= u1.max_energy
            and h1 <= u1.max_hover_time
            and f1 <= u1.max_compute
            and u1.is_compatible(t2)
        )
        cap2 = (
            e2 <= u2.max_energy
            and h2 <= u2.max_hover_time
            and f2 <= u2.max_compute
            and u2.is_compatible(t1)
        )
        return cap1 and cap2

    swaps_done = 0
    improved   = True

    while improved and swaps_done < max_swaps:
        improved = False

        for i, u1 in enumerate(uavs):
            for u2 in uavs[i + 1:]:

                for t1 in list(u1.assigned_tasks):
                    for t2 in list(u2.assigned_tasks):

                        tv_before = region_tv(u1) + region_tv(u2)

                        # Tentative swap
                        u1.assigned_tasks.remove(t1)
                        u2.assigned_tasks.remove(t2)
                        u1.assigned_tasks.append(t2)
                        u2.assigned_tasks.append(t1)

                        tv_after = region_tv(u1) + region_tv(u2)

                        cap_ok = swap_feasible(u1, t1, u2, t2)

                        if tv_after < tv_before and cap_ok:
                            # Accept swap
                            swaps_done += 1
                            improved    = True
                        else:
                            # Revert
                            u1.assigned_tasks.remove(t2)
                            u2.assigned_tasks.remove(t1)
                            u1.assigned_tasks.append(t1)
                            u2.assigned_tasks.append(t2)


# ----------------------------------------------------------
# LAGRANGE MULTIPLIER UPDATE  (eq 8)
#
# μ_{u,k} ← [μ_{u,k} + ρ(Σ_i ω_{k,i}·x_ui − C_{u,k})]+
# ----------------------------------------------------------

def update_lagrange_multipliers(uav):
    """
    Update all three Lagrange multipliers for UAV u.
    Usage includes travel overhead for energy / hover.
    """

    total_energy = sum(
        t.energy_cost +
        ENERGY_PER_METER * math.hypot(uav.x - t.x, uav.y - t.y)
        for t in uav.assigned_tasks
    )
    total_hover = sum(
        t.hover_time +
        math.hypot(uav.x - t.x, uav.y - t.y) / UAV_SPEED
        for t in uav.assigned_tasks
    )
    total_compute = sum(t.compute_load for t in uav.assigned_tasks)

    # Energy multiplier
    uav.mu_energy = max(
        0.0,
        uav.mu_energy + RHO * (total_energy - uav.max_energy)
    )

    # Hover-time multiplier
    uav.mu_hover = max(
        0.0,
        uav.mu_hover + RHO * (total_hover - uav.max_hover_time)
    )

    # Compute multiplier
    uav.mu_compute = max(
        0.0,
        uav.mu_compute + RHO * (total_compute - uav.max_compute)
    )


# ----------------------------------------------------------
# MAIN PARTITIONING ALGORITHM  (Algorithm in D-module)
# ----------------------------------------------------------

def assign_tasks(task_list, uavs):
    """
    Capacity-Constrained Power-Diagram Partitioning.

    Returns:
        uavs  – each with .assigned_tasks populated
        unassigned_tasks – tasks that could not be allocated
    """

    # ======================================================
    # INITIALISATION
    # ======================================================
    for uav in uavs:
        uav.clear_tasks()
        uav.mu_energy  = 0.0
        uav.mu_hover   = 0.0
        uav.mu_compute = 0.0

    # ======================================================
    # ITERATIVE POWER-DIAGRAM OPTIMISATION
    # ======================================================
    for iteration in range(ITERATIONS):

        # Clear assignments for fresh iteration
        for uav in uavs:
            uav.clear_tasks()

        # -----------------------------------------------
        # Sort tasks by priority (critical first),
        # then by total workload descending  (paper: high-
        # priority tasks allocated first)
        # -----------------------------------------------
        sorted_tasks = sorted(
            task_list,
            key=lambda t: (
                t.priority,
                -(t.energy_cost + t.hover_time + t.compute_load)
            )
        )

        unassigned_tasks = []

        # ================================================
        # TASK ASSIGNMENT
        # ================================================
        for task in sorted_tasks:

            feasible_candidates = []

            for uav in uavs:

                if not uav.active:
                    continue

                if not is_feasible(uav, task):
                    continue

                # Generalized cost  δ_ui  (eq 7)
                cost = generalized_cost(uav, task)

                # TV regularisation (compactness)
                cost += compactness_penalty(uav, task)

                feasible_candidates.append((cost, uav))

            # Assign to minimum-cost feasible UAV
            if feasible_candidates:
                feasible_candidates.sort(key=lambda x: x[0])
                best_uav = feasible_candidates[0][1]
                best_uav.assigned_tasks.append(task)
            else:
                unassigned_tasks.append(task)

        # ================================================
        # LAGRANGE MULTIPLIER UPDATE  (eq 8)
        # ================================================
        for uav in uavs:
            update_lagrange_multipliers(uav)

    # ======================================================
    # LOCAL 1-SWAP REFINEMENT  (TV term improvement)
    # ======================================================
    local_swap_refinement(uavs)

    # ======================================================
    # FINAL REPORT
    # ======================================================
    _print_partitioning_summary(uavs, unassigned_tasks)

    return uavs, unassigned_tasks


# ----------------------------------------------------------
# DYNAMIC RE-PARTITIONING  (hysteresis-gated, post Section 5)
# ----------------------------------------------------------

def repartition_with_hysteresis(
    task_list,
    uavs,
    prev_objective,
    hysteresis_threshold=5.0,
    max_reassignments=10,
):
    """
    Re-runs partitioning only if the predicted objective
    improvement exceeds the hysteresis threshold ε.

    Uses a budget B = max_reassignments to limit churn.
    Returns (uavs, new_objective, reassigned_count).
    """

    # Compute current objective
    def objective(uavs_):
        total = 0.0
        for uav in uavs_:
            for t in uav.assigned_tasks:
                dist = math.hypot(uav.x - t.x, uav.y - t.y)
                total += ALPHA * dist - GAMMA * (
                    100 if t.priority == 1 else
                    60  if t.priority == 2 else 20
                )
        return total

    current_obj = objective(uavs)

    # Compute tentative new objective (single pass)
    test_uavs = [_clone_uav(u) for u in uavs]
    assign_tasks(task_list, test_uavs)
    new_obj = objective(test_uavs)

    delta_j = abs(new_obj - current_obj)

    if delta_j < hysteresis_threshold:
        return uavs, current_obj, 0   # no re-partition needed

    # Accept new partition, but cap reassignments
    reassigned = 0
    for u_old, u_new in zip(uavs, test_uavs):
        old_set = set(t.task_id for t in u_old.assigned_tasks)
        new_set = set(t.task_id for t in u_new.assigned_tasks)
        changes = len(old_set.symmetric_difference(new_set)) // 2
        if reassigned + changes > max_reassignments:
            break
        u_old.assigned_tasks = u_new.assigned_tasks
        reassigned += changes

    return uavs, new_obj, reassigned


def _clone_uav(uav):
    """Shallow clone for hysteresis testing."""
    from uav import UAV
    u2 = UAV(
        uav.uav_id,
        uav.x, uav.y,
        uav.uav_type,
        uav.max_energy,
        uav.max_hover_time,
        uav.max_compute,
    )
    u2.mu_energy  = uav.mu_energy
    u2.mu_hover   = uav.mu_hover
    u2.mu_compute = uav.mu_compute
    return u2


# ----------------------------------------------------------
# HELPER: PRINT SUMMARY
# ----------------------------------------------------------

def _print_partitioning_summary(uavs, unassigned):

    print("\n=== REGION PARTITIONING SUMMARY ===")

    for uav in uavs:

        te = sum(
            t.energy_cost +
            ENERGY_PER_METER * math.hypot(uav.x - t.x, uav.y - t.y)
            for t in uav.assigned_tasks
        )
        th = sum(t.hover_time   for t in uav.assigned_tasks)
        tf = sum(t.compute_load for t in uav.assigned_tasks)

        hp = sum(1 for t in uav.assigned_tasks if t.priority == 1)

        print(
            f"  UAV {uav.uav_id:02d} (type {uav.uav_type:+d}) | "
            f"tasks={len(uav.assigned_tasks):3d} "
            f"(hi-pri={hp}) | "
            f"E={te:6.1f}/{uav.max_energy:6.1f}J  "
            f"H={th:5.1f}/{uav.max_hover_time:5.1f}s  "
            f"F={tf:5.1f}/{uav.max_compute:5.1f}GHz·s"
        )

    if unassigned:
        print(
            f"\n  ⚠  {len(unassigned)} tasks could not be assigned"
            f" (insufficient fleet capacity)"
        )

    print()