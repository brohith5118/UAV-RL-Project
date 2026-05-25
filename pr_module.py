# =========================================================
# PR-MODULE  –  SOM-based Pre-Assignment & Re-Assignment
#
# Improved + Fixed Version
#
# Major fixes:
#   ✔ Uses current_x/current_y consistently
#   ✔ Proper residual-resource rollback
#   ✔ Sequential feasibility checks
#   ✔ Stable SOM updates
#   ✔ Correct resource accounting
#   ✔ Avoids infeasible over-assignment
#   ✔ Dynamic-event safe reassignment
#   ✔ Better assignment balancing
#   ✔ Energy-aware matching
# =========================================================

import math
import random
import numpy as np

from config import (
    SOM_ITERATIONS,
    SOM_LEARN_RATE,
    C_PHI,
    C_RES,
    C_S,
    C_TIME,
    C_COMP,
    UAV_SPEED,
    ENERGY_PER_METER,
)


# =========================================================
# BASIC HELPERS
# =========================================================

def euclidean(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def travel_distance(uav, task):
    return euclidean(
        uav.current_x,
        uav.current_y,
        task.x,
        task.y
    )


def travel_time(uav, task):
    return travel_distance(uav, task) / UAV_SPEED


def travel_energy(uav, task):
    return travel_distance(uav, task) * ENERGY_PER_METER


# =========================================================
# RESOURCE FEASIBILITY
# =========================================================

def capability_mismatch(task, uav):
    """
    D_phi (eq 19)
    """

    if not uav.is_compatible(task):
        return float("inf")

    return 0.0


def time_margin_penalty(task, uav, base_x=0.0, base_y=0.0):
    """
    Remaining hover-time feasibility.
    """

    d_task = travel_distance(uav, task)

    d_base = euclidean(
        task.x,
        task.y,
        base_x,
        base_y
    )

    total_travel = (d_task + d_base) / UAV_SPEED

    diff = (
        uav.remaining_hover_time
        - total_travel
        - task.hover_time
    )

    if diff < 0:
        return float("inf")

    return math.exp(-C_TIME * diff)


def compute_margin_penalty(task, uav):

    diff = (
        uav.remaining_compute
        - task.compute_load
    )

    if diff < 0:
        return float("inf")

    return math.exp(-C_COMP * diff)


def energy_margin_penalty(task, uav):

    total_needed = (
        task.energy_cost
        + travel_energy(uav, task)
    )

    diff = (
        uav.remaining_energy
        - total_needed
    )

    if diff < 0:
        return float("inf")

    return math.exp(-0.01 * diff)


def is_feasible(task, uav):

    if capability_mismatch(task, uav) == float("inf"):
        return False

    if time_margin_penalty(task, uav) == float("inf"):
        return False

    if compute_margin_penalty(task, uav) == float("inf"):
        return False

    if energy_margin_penalty(task, uav) == float("inf"):
        return False

    return True


# =========================================================
# MATCHING DISTANCE
# =========================================================

def spatial_distance(task, uav):
    """
    Uses CURRENT UAV position.
    """

    return euclidean(
        task.x,
        task.y,
        uav.current_x,
        uav.current_y
    )


def matching_distance(task, uav):

    d_phi = capability_mismatch(task, uav)

    if d_phi == float("inf"):
        return float("inf")

    d_time = time_margin_penalty(task, uav)

    if d_time == float("inf"):
        return float("inf")

    d_comp = compute_margin_penalty(task, uav)

    if d_comp == float("inf"):
        return float("inf")

    d_energy = energy_margin_penalty(task, uav)

    if d_energy == float("inf"):
        return float("inf")

    d_spatial = spatial_distance(task, uav)

    # Load balancing term
    load_penalty = len(uav.assigned_tasks) * 5.0

    return (
        d_spatial
        + C_PHI * d_phi
        + C_RES * (
            d_time
            + d_comp
            + d_energy
        )
        + load_penalty
    )


# =========================================================
# SOM NEIGHBOURHOOD
# =========================================================

def neighbourhood(uav, winner_uav, all_uavs):

    if uav.uav_id == winner_uav.uav_id:
        return 1.0

    if abs(uav.uav_type - winner_uav.uav_type) > 1:
        return 0.0

    idx_u = next(
        i for i, v in enumerate(all_uavs)
        if v.uav_id == uav.uav_id
    )

    idx_w = next(
        i for i, v in enumerate(all_uavs)
        if v.uav_id == winner_uav.uav_id
    )

    s = abs(idx_u - idx_w)

    return math.exp(-s / C_S)


# =========================================================
# FEATURE VECTORS
# =========================================================

def uav_feature(uav):

    return np.array([
        uav.current_x,
        uav.current_y,
        float(uav.uav_type),
        uav.remaining_hover_time,
        uav.remaining_compute,
        uav.remaining_energy,
    ], dtype=np.float64)


def task_feature(task):

    return np.array([
        task.x,
        task.y,
        float(task.task_type),
        task.hover_time,
        task.compute_load,
        task.energy_cost,
    ], dtype=np.float64)


# =========================================================
# RESOURCE UPDATE
# =========================================================

def assign_task_to_uav(task, uav):

    uav.assigned_tasks.append(task)

    uav.consume_resources(task)

    # Move UAV virtual current position
    uav.current_x = task.x
    uav.current_y = task.y

    task.assigned_uav = uav


def rollback_task_from_uav(task, uav):

    if task not in uav.assigned_tasks:
        return

    uav.assigned_tasks.remove(task)

    # Full rollback approximation
    uav.reset_resources()

    uav.current_x = uav.x
    uav.current_y = uav.y

    remaining = list(uav.assigned_tasks)

    uav.assigned_tasks = []

    for t in remaining:
        assign_task_to_uav(t, uav)

    task.assigned_uav = None


# =========================================================
# CORE SOM ASSIGNMENT
# =========================================================

def som_assign(tasks, uavs):

    if not tasks or not uavs:
        return {}

    active_uavs = [
        u for u in uavs
        if u.active
    ]

    if not active_uavs:
        return {
            t.task_id: None
            for t in tasks
        }

    # =====================================================
    # RESET TEMPORARY STATE
    # =====================================================

    for uav in active_uavs:

        uav.current_x = uav.x
        uav.current_y = uav.y

    # =====================================================
    # SOM WEIGHTS
    # =====================================================

    uav_features = {
        u.uav_id: uav_feature(u)
        for u in active_uavs
    }

    learn_rate = SOM_LEARN_RATE

    decay = learn_rate / max(SOM_ITERATIONS, 1)

    # =====================================================
    # TRAINING
    # =====================================================

    for iteration in range(SOM_ITERATIONS):

        shuffled = list(tasks)

        random.shuffle(shuffled)

        for task in shuffled:

            tf = task_feature(task)

            best_uav = None
            best_dist = float("inf")

            # ---------------------------------------------
            # Find BMU
            # ---------------------------------------------

            for uav in active_uavs:

                d = matching_distance(task, uav)

                if d < best_dist:
                    best_dist = d
                    best_uav = uav

            if best_uav is None:
                continue

            # ---------------------------------------------
            # SOM UPDATE
            # ---------------------------------------------

            for uav in active_uavs:

                n = neighbourhood(
                    uav,
                    best_uav,
                    active_uavs
                )

                if n <= 0:
                    continue

                feat = uav_features[uav.uav_id]

                feat += (
                    n
                    * learn_rate
                    * (tf - feat)
                )

                uav_features[uav.uav_id] = feat

        learn_rate = max(
            0.01,
            learn_rate - decay
        )

    # =====================================================
    # FINAL GREEDY ASSIGNMENT
    # =====================================================

    assignment = {}

    # Important:
    # high-priority first
    sorted_tasks = sorted(
        tasks,
        key=lambda t: (
            t.priority,
            -(t.compute_load + t.energy_cost)
        )
    )

    for task in sorted_tasks:

        best_uav = None
        best_dist = float("inf")

        for uav in active_uavs:

            if not is_feasible(task, uav):
                continue

            d = matching_distance(task, uav)

            if d < best_dist:
                best_dist = d
                best_uav = uav

        assignment[task.task_id] = best_uav

        if best_uav is not None:
            assign_task_to_uav(task, best_uav)

    return assignment


# =========================================================
# PRE-ASSIGNMENT
# =========================================================

def preassign(tasks, uavs):

    # Reset everything
    for uav in uavs:

        uav.clear_tasks()

        uav.reset_resources()

        uav.current_x = uav.x
        uav.current_y = uav.y

    for task in tasks:
        task.assigned_uav = None

    som_assign(tasks, uavs)

    _print_assignment_summary(
        "PRE-ASSIGNMENT",
        uavs,
        tasks
    )

    return uavs


# =========================================================
# NEW TASK INSERTION
# =========================================================

def reassign_new_tasks(new_tasks, uavs):

    for task in new_tasks:
        task.assigned_uav = None

    som_assign(new_tasks, uavs)

    print(
        f"\n[PR] Inserted {len(new_tasks)} new tasks."
    )

    return uavs


# =========================================================
# LOCATION UPDATE
# =========================================================

def reassign_after_location_update(
    updated_tasks,
    uavs
):

    for task in updated_tasks:

        old_uav = task.assigned_uav

        if old_uav is not None:

            rollback_task_from_uav(
                task,
                old_uav
            )

    som_assign(updated_tasks, uavs)

    print(
        f"\n[PR] Reassigned "
        f"{len(updated_tasks)} updated tasks."
    )

    return uavs


# =========================================================
# UAV FAILURE
# =========================================================

def reassign_after_uav_failure(
    failed_uav,
    uavs
):

    failed_uav.active = False

    orphaned = [
        t for t in failed_uav.assigned_tasks
        if not t.completed
    ]

    failed_uav.assigned_tasks = []

    failed_uav.reset_resources()

    failed_uav.current_x = failed_uav.x
    failed_uav.current_y = failed_uav.y

    for task in orphaned:
        task.assigned_uav = None

    active_uavs = [
        u for u in uavs
        if u.active
    ]

    som_assign(orphaned, active_uavs)

    assigned = sum(
        1 for t in orphaned
        if t.assigned_uav is not None
    )

    print(
        f"\n[PR] UAV {failed_uav.uav_id} failed "
        f"→ reassigned {assigned}/{len(orphaned)} tasks."
    )

    return uavs


# =========================================================
# TASK CANCELLATION
# =========================================================

def cancel_tasks(cancelled_tasks, uavs):

    cancelled_ids = {
        t.task_id
        for t in cancelled_tasks
    }

    for uav in uavs:

        to_remove = [
            t for t in uav.assigned_tasks
            if t.task_id in cancelled_ids
        ]

        for task in to_remove:

            rollback_task_from_uav(
                task,
                uav
            )

    print(
        f"\n[PR] Cancelled "
        f"{len(cancelled_tasks)} tasks."
    )

    return uavs


# =========================================================
# PRINT SUMMARY
# =========================================================

def _print_assignment_summary(
    label,
    uavs,
    all_tasks
):

    assigned = sum(
        1 for t in all_tasks
        if t.assigned_uav is not None
    )

    print(f"\n=== {label} ===")

    print(
        f"Assigned: "
        f"{assigned}/{len(all_tasks)}"
    )

    for uav in uavs:

        if not uav.active:
            continue

        hi = sum(
            1 for t in uav.assigned_tasks
            if t.priority == 1
        )

        print(
            f"UAV {uav.uav_id:02d} "
            f"(type {uav.uav_type:+d}) | "
            f"tasks={len(uav.assigned_tasks):2d} | "
            f"hi-pri={hi} | "
            f"rem-E={uav.remaining_energy:.1f} | "
            f"rem-H={uav.remaining_hover_time:.1f} | "
            f"rem-C={uav.remaining_compute:.1f}"
        )

    print()