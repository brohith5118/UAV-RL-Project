# =========================================================
# PR-MODULE  –  SOM-based Pre-Assignment & Re-Assignment
#
# Implements Algorithm 1 from the paper (Section
# "Network structure and input representation", eq 15–26).
#
# Input feature vectors:
#   INFO_TASK_i = (POS_i, ϕ_i, RES_i)          (eq 15)
#   INFO_UAV_u  = (POS_u, ψ_u, RES_u)          (eq 16)
#
# Matching distance:
#   D(i,u) = D_p + c_ϕ·D_ϕ + c_RES·D_RES       (eq 17)
#
# Neighbourhood update (eq 25–26) ensures topology
# coherence across UAV types.
#
# Dynamic events handled:
#   (1) New task insertion
#   (2) Task location update
#   (3) UAV failure / task redistribution
#   (4) Task cancellation
# =========================================================

import math
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


# ----------------------------------------------------------
# MATCHING DISTANCE  D(i,u)  (eq 17–24)
# ----------------------------------------------------------

def spatial_distance(task, uav):
    """D_p = (X_i − X_u)² + (Y_i − Y_u)²  (eq 18)"""
    return (task.x - uav.x) ** 2 + (task.y - uav.y) ** 2


def capability_mismatch(task, uav):
    """
    D_ϕ (eq 19):
      0       if |ψ_u − ϕ_i| ≤ 1  (compatible)
      ∞       otherwise
    """
    if abs(uav.uav_type - task.task_type) <= 1:
        return 0.0
    return float('inf')


def time_margin_penalty(task, uav, base_x=0.0, base_y=0.0):
    """
    Δ_time  (eq 20–22):
      diff_time = T^re_u − (d(task,uav)+d(task,base))/v − t^req_i
      penalty   = exp(−c_time · diff_time)  if diff_time ≥ 0
                = ∞                          otherwise
    """
    d_to_task = math.hypot(task.x - uav.x, task.y - uav.y)
    d_to_base = math.hypot(task.x - base_x, task.y - base_y)
    travel    = (d_to_task + d_to_base) / UAV_SPEED
    diff_time = uav.remaining_hover_time - travel - task.hover_time

    if diff_time < 0:
        return float('inf')
    return math.exp(-C_TIME * diff_time)


def compute_margin_penalty(task, uav):
    """
    Δ_comp  (eq 23–24):
      diff_comp = C^re_u − c^req_i
      penalty   = exp(−c_comp · diff_comp)  if diff_comp ≥ 0
                = ∞                          otherwise
    """
    diff_comp = uav.remaining_compute - task.compute_load
    if diff_comp < 0:
        return float('inf')
    return math.exp(-C_COMP * diff_comp)


def matching_distance(task, uav, base_x=0.0, base_y=0.0):
    """
    Full matching distance D(i,u)  (eq 17).
    Returns ∞ if the pair is infeasible.
    """
    d_phi = capability_mismatch(task, uav)
    if d_phi == float('inf'):
        return float('inf')

    d_p   = spatial_distance(task, uav)
    d_res = (
        time_margin_penalty(task, uav, base_x, base_y)
        + compute_margin_penalty(task, uav)
    )

    if d_res == float('inf'):
        return float('inf')

    return d_p + C_PHI * d_phi + C_RES * d_res


# ----------------------------------------------------------
# SOM NEIGHBOURHOOD FUNCTION  n_{u,u*}  (eq 25)
# ----------------------------------------------------------

def neighbourhood(uav, winner_uav, all_uavs):
    """
    n_{u,u*} (eq 25):
      1                        if u == u*
      exp(−S_{u,u*} / c_s)    if |ψ_u − ψ_u*| ≤ 1
      0                        otherwise

    S_{u,u*} is the index distance in the UAV list (proxy
    for SOM grid distance).
    """
    if uav.uav_id == winner_uav.uav_id:
        return 1.0

    if abs(uav.uav_type - winner_uav.uav_type) > 1:
        return 0.0

    idx_u      = next(i for i, v in enumerate(all_uavs)
                      if v.uav_id == uav.uav_id)
    idx_winner = next(i for i, v in enumerate(all_uavs)
                      if v.uav_id == winner_uav.uav_id)
    s = abs(idx_u - idx_winner)
    return math.exp(-s / C_S)


# ----------------------------------------------------------
# SOM FEATURE VECTORS  (residual-based, updated each round)
# ----------------------------------------------------------

def uav_feature(uav):
    """INFO_UAV_u = (X_u, Y_u, ψ_u, T^re_u, C^re_u)"""
    return np.array([
        uav.x,
        uav.y,
        float(uav.uav_type),
        uav.remaining_hover_time,
        uav.remaining_compute,
    ], dtype=np.float64)


def task_feature(task):
    """INFO_TASK_i = (X_i, Y_i, ϕ_i, t^req_i, c^req_i)"""
    return np.array([
        task.x,
        task.y,
        float(task.task_type),
        task.hover_time,
        task.compute_load,
    ], dtype=np.float64)


# ----------------------------------------------------------
# CORE SOM ASSIGNMENT  (Algorithm 1)
# ----------------------------------------------------------

def som_assign(tasks, uavs, base_x=0.0, base_y=0.0):
    """
    Run SOM pre/re-assignment for a list of *tasks* over
    the active *uavs*.

    Returns a dict  {task_id: uav}  with the best feasible
    assignment for every task.  Tasks that remain infeasible
    for all UAVs are mapped to None.
    """

    if not tasks or not uavs:
        return {}

    active_uavs = [u for u in uavs if u.active]
    if not active_uavs:
        return {t.task_id: None for t in tasks}

    # Initialise UAV feature vectors (SOM weights)
    uav_features = {u.uav_id: uav_feature(u) for u in active_uavs}

    learn_rate = SOM_LEARN_RATE
    lr_decay   = learn_rate / SOM_ITERATIONS

    # -------------------------------------------------------
    # SOM TRAINING ITERATIONS
    # -------------------------------------------------------
    for r in range(SOM_ITERATIONS):

        # Present tasks in random order each iteration
        shuffled = list(tasks)
        np.random.default_rng(r).shuffle(shuffled)

        for task in shuffled:

            tf = task_feature(task)

            # Find best-matching UAV (BMU) = argmin D(i,u)
            best_dist  = float('inf')
            best_uav   = None

            for uav in active_uavs:
                d = matching_distance(task, uav, base_x, base_y)
                if d < best_dist:
                    best_dist = d
                    best_uav  = uav

            if best_uav is None:
                continue

            # Neighbourhood update  (eq 26)
            for uav in active_uavs:
                n = neighbourhood(uav, best_uav, active_uavs)
                if n == 0.0:
                    continue
                feat   = uav_features[uav.uav_id]
                # Move UAV feature vector toward task
                feat  += n * learn_rate * (tf - feat)
                uav_features[uav.uav_id] = feat

        learn_rate = max(0.01, learn_rate - lr_decay)

    # -------------------------------------------------------
    # FINAL ASSIGNMENT  – bind each task to best feasible UAV
    # -------------------------------------------------------
    assignment = {}

    for task in tasks:
        best_dist = float('inf')
        best_uav  = None

        for uav in active_uavs:
            d = matching_distance(task, uav, base_x, base_y)
            if d < best_dist:
                best_dist = d
                best_uav  = uav

        assignment[task.task_id] = best_uav

    return assignment


# ----------------------------------------------------------
# PR-MODULE PUBLIC API
# ----------------------------------------------------------

def preassign(tasks, uavs, base_x=0.0, base_y=0.0):
    """
    Initial pre-assignment of *tasks* to *uavs*.
    Clears existing assignments first, then runs SOM.

    Returns updated uavs (each with .assigned_tasks set).
    """
    for uav in uavs:
        uav.clear_tasks()
        uav.reset_resources()

    assignment = som_assign(tasks, uavs, base_x, base_y)

    for task in tasks:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav

    _print_assignment_summary("PRE-ASSIGNMENT", uavs, tasks)
    return uavs


def reassign_new_tasks(new_tasks, uavs, base_x=0.0, base_y=0.0):
    """
    Dynamic event (1): Insert new urgent tasks mid-mission.
    Only new_tasks are reassigned; existing assignments kept.
    """
    assignment = som_assign(new_tasks, uavs, base_x, base_y)

    for task in new_tasks:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav

    print(f"\n[PR] Inserted {len(new_tasks)} new tasks.")
    return uavs


def reassign_after_location_update(updated_tasks, uavs,
                                   base_x=0.0, base_y=0.0):
    """
    Dynamic event (2): Task location changed – reassign only
    the affected tasks (remove from old UAV, re-run SOM).
    """
    for task in updated_tasks:
        old_uav = task.assigned_uav
        if old_uav is not None and task in old_uav.assigned_tasks:
            old_uav.assigned_tasks.remove(task)
            # Return resources
            old_uav.remaining_energy     += task.energy_cost
            old_uav.remaining_hover_time += task.hover_time
            old_uav.remaining_compute    += task.compute_load

    assignment = som_assign(updated_tasks, uavs, base_x, base_y)

    for task in updated_tasks:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav

    print(f"\n[PR] Re-assigned {len(updated_tasks)} location-updated tasks.")
    return uavs


def reassign_after_uav_failure(failed_uav, uavs,
                               base_x=0.0, base_y=0.0):
    """
    Dynamic event (3): A UAV fails mid-mission.
    Redistribute its remaining (incomplete) tasks.
    """
    failed_uav.active = False
    orphaned = [
        t for t in failed_uav.assigned_tasks
        if not t.completed
    ]
    failed_uav.assigned_tasks = []

    if not orphaned:
        print(f"\n[PR] UAV {failed_uav.uav_id} failed – no orphaned tasks.")
        return uavs

    active_uavs = [u for u in uavs if u.active]
    assignment  = som_assign(orphaned, active_uavs, base_x, base_y)

    redistributed = 0
    for task in orphaned:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav
            redistributed += 1

    print(
        f"\n[PR] UAV {failed_uav.uav_id} failed – "
        f"redistributed {redistributed}/{len(orphaned)} tasks."
    )
    return uavs


def cancel_tasks(cancelled_tasks, uavs):
    """
    Dynamic event (4): Tasks cancelled – remove from UAV
    lists and return resources.
    """
    cancelled_ids = {t.task_id for t in cancelled_tasks}

    for uav in uavs:
        to_remove = [
            t for t in uav.assigned_tasks
            if t.task_id in cancelled_ids
        ]
        for t in to_remove:
            uav.assigned_tasks.remove(t)
            uav.remaining_energy     += t.energy_cost
            uav.remaining_hover_time += t.hover_time
            uav.remaining_compute    += t.compute_load

    print(f"\n[PR] Cancelled {len(cancelled_tasks)} tasks.")
    return uavs


# ----------------------------------------------------------
# HELPER: PRINT SUMMARY
# ----------------------------------------------------------

def _print_assignment_summary(label, uavs, all_tasks):
    assigned = sum(
        1 for t in all_tasks if t.assigned_uav is not None
    )
    print(f"\n=== {label} ===")
    print(f"  Tasks assigned: {assigned}/{len(all_tasks)}")
    for uav in uavs:
        if not uav.active:
            continue
        hi = sum(1 for t in uav.assigned_tasks if t.priority == 1)
        print(
            f"  UAV {uav.uav_id:02d} (type {uav.uav_type:+d}) → "
            f"{len(uav.assigned_tasks)} tasks  (hi-pri={hi})  "
            f"rem-compute={uav.remaining_compute:.1f}  "
            f"rem-hover={uav.remaining_hover_time:.0f}s"
        )
    print()