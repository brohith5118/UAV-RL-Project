# =========================================================
# SCHEDULER  –  D-MODULE
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
    MAP_WIDTH,
    MAP_HEIGHT,
)


# ----------------------------------------------------------
# DISTANCE HELPERS
# ----------------------------------------------------------

def euclidean(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


# ----------------------------------------------------------
# SEQUENTIAL ROUTE RESOURCE ESTIMATION
# ----------------------------------------------------------

def estimate_route_cost(uav, tasks):

    prev_x = uav.x
    prev_y = uav.y

    total_energy = 0.0
    total_hover = 0.0
    total_compute = 0.0

    for task in tasks:

        dist = euclidean(prev_x, prev_y, task.x, task.y)

        travel_energy = dist * ENERGY_PER_METER
        travel_time = dist / UAV_SPEED

        total_energy += task.energy_cost + travel_energy
        total_hover += task.hover_time + travel_time
        total_compute += task.compute_load

        prev_x = task.x
        prev_y = task.y

    return total_energy, total_hover, total_compute


# ----------------------------------------------------------
# REGION CENTROID UPDATE
# ----------------------------------------------------------

def update_region_centroids(uavs):

    for uav in uavs:

        if not uav.assigned_tasks:
            continue

        n = len(uav.assigned_tasks)

        uav.region_x = (
            sum(t.x for t in uav.assigned_tasks) / n
        )

        uav.region_y = (
            sum(t.y for t in uav.assigned_tasks) / n
        )


# ----------------------------------------------------------
# FEASIBILITY CHECK
# ----------------------------------------------------------

def is_feasible(uav, task):

    if not uav.is_compatible(task):
        return False

    tentative_tasks = uav.assigned_tasks + [task]

    total_energy, total_hover, total_compute = (
        estimate_route_cost(uav, tentative_tasks)
    )

    if total_energy > uav.max_energy:
        return False

    if total_hover > uav.max_hover_time:
        return False

    if total_compute > uav.max_compute:
        return False

    return True


# ----------------------------------------------------------
# GENERALIZED COST FUNCTION
# ----------------------------------------------------------

def generalized_cost(uav, task, avg_tasks):

    distance = euclidean(
        uav.region_x,
        uav.region_y,
        task.x,
        task.y
    )

    # NORMALIZED DISTANCE
    map_diag = math.hypot(MAP_WIDTH, MAP_HEIGHT)
    norm_distance = distance / map_diag

    # NORMALIZED PRIORITY
    priority_reward = {
        1: 1.0,
        2: 0.6,
        3: 0.2
    }[task.priority]

    load_ratio = (len(uav.assigned_tasks) / max(avg_tasks, 1))

    load_penalty = load_ratio ** 2

    total_energy, total_hover, total_compute = (
        estimate_route_cost(
            uav,
            uav.assigned_tasks
        )
    )

    energy_ratio = total_energy / max(uav.max_energy, 1)

    hover_ratio = total_hover / max(uav.max_hover_time, 1)

    compute_ratio = (
        total_compute / max(uav.max_compute, 1)
        if uav.max_compute > 0
        else 0
    )

    resource_penalty = (
        energy_ratio
        + hover_ratio
        + compute_ratio
    )

    # LAGRANGE PENALTY
    lagrange_penalty = (
        uav.mu_energy * energy_ratio
        + uav.mu_hover * hover_ratio
        + uav.mu_compute * compute_ratio
    )


    total_cost = (
        ALPHA * norm_distance
        + 2.0 * load_penalty
        + 2.0 * resource_penalty
        + lagrange_penalty
        - GAMMA * priority_reward
    )

    return total_cost


# ----------------------------------------------------------
# TV REGULARIZATION
# ----------------------------------------------------------

def compactness_penalty(uav, task):

    if not uav.assigned_tasks:
        return 0.0

    n = len(uav.assigned_tasks)

    cx = sum(t.x for t in uav.assigned_tasks) / n
    cy = sum(t.y for t in uav.assigned_tasks) / n

    centroid_dist = euclidean(
        task.x,
        task.y,
        cx,
        cy
    )

    return LAMBDA_TV * centroid_dist


# ----------------------------------------------------------
# LAGRANGE MULTIPLIER UPDATE
# ----------------------------------------------------------

def update_lagrange_multipliers(uav):

    total_energy, total_hover, total_compute = (
        estimate_route_cost(
            uav,
            uav.assigned_tasks
        )
    )

    uav.mu_energy = max(
        0.0,
        uav.mu_energy +
        RHO * (total_energy - uav.max_energy)
    )

    uav.mu_hover = max(
        0.0,
        uav.mu_hover +
        RHO * (total_hover - uav.max_hover_time)
    )

    uav.mu_compute = max(
        0.0,
        uav.mu_compute +
        RHO * (total_compute - uav.max_compute)
    )


# ----------------------------------------------------------
# GREEDY WARM START ROUTE
# ----------------------------------------------------------

def nearest_neighbor_order(uav):

    if not uav.assigned_tasks:
        return

    remaining = uav.assigned_tasks[:]
    ordered = []

    current_x = uav.x
    current_y = uav.y

    while remaining:

        nearest = min(
            remaining,
            key=lambda t: euclidean(
                current_x,
                current_y,
                t.x,
                t.y
            )
        )

        ordered.append(nearest)

        current_x = nearest.x
        current_y = nearest.y

        remaining.remove(nearest)

    uav.assigned_tasks = ordered


# ----------------------------------------------------------
# MAIN PARTITIONING ALGORITHM
# ----------------------------------------------------------

def assign_tasks(task_list, uavs):

    # INITIAL RESET ONLY ONCE
    for uav in uavs:

        uav.clear_tasks()

        uav.mu_energy = 0.0
        uav.mu_hover = 0.0
        uav.mu_compute = 0.0

        uav.region_x = uav.x
        uav.region_y = uav.y

    sorted_tasks = sorted(
        task_list,
        key=lambda t: (
            t.priority,
            -(t.energy_cost +
              t.hover_time +
              t.compute_load)
        )
    )

    unassigned_tasks = []

    for iteration in range(ITERATIONS):

        # STORE OLD ASSIGNMENTS
        old_assignments = {
            u.uav_id: set(
                t.task_id for t in u.assigned_tasks
            )
            for u in uavs
        }

        # CLEAR FOR REFINEMENT
        for uav in uavs:
            uav.clear_tasks()

        unassigned_tasks = []

        avg_tasks = (
            len(task_list) / len(uavs)
            if uavs else 1
        )

        # ASSIGN TASKS
        for task in sorted_tasks:

            feasible_candidates = []

            for uav in uavs:

                if not uav.active:
                    continue

                if not is_feasible(uav, task):
                    continue

                cost = generalized_cost(
                    uav,
                    task,
                    avg_tasks
                )

                cost += compactness_penalty(
                    uav,
                    task
                )

                feasible_candidates.append(
                    (cost, uav)
                )

            if feasible_candidates:

                feasible_candidates.sort(
                    key=lambda x: x[0]
                )

                best_uav = feasible_candidates[0][1]

                best_uav.assigned_tasks.append(task)

            else:
                unassigned_tasks.append(task)

        # UPDATE CENTROIDS ONCE PER ITERATION
        update_region_centroids(uavs)

        # UPDATE LAGRANGE MULTIPLIERS
        for uav in uavs:
            update_lagrange_multipliers(uav)

        # CONVERGENCE CHECK
        converged = True

        for uav in uavs:

            new_set = set(
                t.task_id
                for t in uav.assigned_tasks
            )

            if new_set != old_assignments[uav.uav_id]:
                converged = False
                break

        if converged:
            print(
                f"[D-Module] Converged "
                f"after {iteration+1} iterations"
            )
            break

    # GREEDY ROUTE INITIALIZATION
    for uav in uavs:
        nearest_neighbor_order(uav)

    _print_partitioning_summary(
        uavs,
        unassigned_tasks
    )

    return uavs, unassigned_tasks


# ----------------------------------------------------------
# PARTITIONING SUMMARY
# ----------------------------------------------------------

def _print_partitioning_summary(
    uavs,
    unassigned
):

    print("\n=== PARTITIONING SUMMARY ===")

    for uav in uavs:

        te, th, tf = estimate_route_cost(
            uav,
            uav.assigned_tasks
        )

        hp = sum(
            1
            for t in uav.assigned_tasks
            if t.priority == 1
        )

        print(
            f"\nUAV {uav.uav_id}"
        )

        print(
            f"Tasks: {len(uav.assigned_tasks)}"
        )

        print(
            f"High Priority: {hp}"
        )

        print(
            f"Energy: "
            f"{te:.2f}/{uav.max_energy:.2f}"
        )

        print(
            f"Hover: "
            f"{th:.2f}/{uav.max_hover_time:.2f}"
        )

        print(
            f"Compute: "
            f"{tf:.2f}/{uav.max_compute:.2f}"
        )

    if unassigned:

        print(
            f"\nUnassigned Tasks: "
            f"{len(unassigned)}"
        )

    print()