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

class TempUAVWeightProxy:
    def __init__(self, uav_feat, uav_id, rem_energy, rem_hover, rem_compute, uav_map):
        self.uav_id = uav_id
        self.x = uav_feat[0]
        self.y = uav_feat[1]
        self.uav_type = int(round(uav_feat[2]))
        
        self.remaining_energy = rem_energy
        self.remaining_hover_time = rem_hover
        self.remaining_compute = rem_compute
        
        orig_uav = uav_map[uav_id]
        self.max_energy = orig_uav.max_energy
        self.max_hover_time = orig_uav.max_hover_time
        self.max_compute = orig_uav.max_compute

    def is_compatible(self, task):
        return abs(self.uav_type - task.task_type) <= 1


def assignment_score_proxy(task, uav, base_x=0.0, base_y=0.0):
    if not uav.is_compatible(task):
        return -float('inf')
        
    dist = math.hypot(uav.x - task.x, uav.y - task.y)
    
    # 1. Spatial closeness
    spatial_closeness = 100.0 / (1.0 + dist)
    
    # 2. Long traversal penalty
    long_traversal_penalty = 6.0 * dist
    
    # 3. Compute feasibility
    if task.compute_load > 0:
        if uav.remaining_compute >= task.compute_load:
            compute_feasibility = 20.0 * (uav.remaining_compute / uav.max_compute)
        else:
            return -float('inf')
    else:
        compute_feasibility = 10.0
        
    # 4. Deadline feasibility
    t_travel = dist / UAV_SPEED
    t_needed = t_travel + task.hover_time
    if uav.remaining_hover_time >= t_needed:
        if t_needed <= task.deadline:
            deadline_feasibility = 20.0 * (1.0 - t_needed / task.deadline)
        else:
            deadline_feasibility = -100.0
    else:
        return -float('inf')
        
    # Check energy constraint
    e_travel = dist * ENERGY_PER_METER
    e_needed = e_travel + task.energy_cost
    if uav.remaining_energy < e_needed:
        return -float('inf')
        
    # 5. Residual energy balancing
    residual_energy_balancing = 20.0 * (uav.remaining_energy / uav.max_energy)
    
    score = (
        spatial_closeness
        + compute_feasibility
        + deadline_feasibility
        + residual_energy_balancing
        - long_traversal_penalty
    )
    return score


def redistribute_load(uavs, base_x=0.0, base_y=0.0):
    active_uavs = [u for u in uavs if u.active]
    if not active_uavs:
        return uavs

    print("\n[PR] Running Dynamic Reallocation Engine for load balancing...")

    def get_utilizations(u):
        total_energy = sum(
            t.energy_cost + ENERGY_PER_METER * math.hypot(u.x - t.x, u.y - t.y)
            for t in u.assigned_tasks
        )
        total_hover = sum(
            t.hover_time + math.hypot(u.x - t.x, u.y - t.y) / UAV_SPEED
            for t in u.assigned_tasks
        )
        total_compute = sum(t.compute_load for t in u.assigned_tasks)
        return total_energy, total_hover, total_compute

    max_passes = 5
    for pass_idx in range(max_passes):
        reallocated_any = False
        overloaded_uavs = []
        
        for u in active_uavs:
            te, th, tf = get_utilizations(u)
            if te > u.max_energy or th > u.max_hover_time or tf > u.max_compute:
                e_ratio = te / u.max_energy
                h_ratio = th / u.max_hover_time
                f_ratio = tf / u.max_compute if u.max_compute > 0 else 0.0
                overflow = max(e_ratio, h_ratio, f_ratio)
                overloaded_uavs.append((overflow, u))
                
        if not overloaded_uavs:
            break
            
        overloaded_uavs.sort(key=lambda x: x[0], reverse=True)
        
        for overflow, u_over in overloaded_uavs:
            tasks_to_migrate = sorted(
                u_over.assigned_tasks,
                key=lambda t: (-t.priority, t.energy_cost + t.hover_time + t.compute_load),
                reverse=True
            )
            
            for task in tasks_to_migrate:
                best_score = -float('inf')
                best_target_uav = None
                
                for u_target in active_uavs:
                    if u_target.uav_id == u_over.uav_id:
                        continue
                    if not u_target.is_compatible(task):
                        continue
                        
                    target_te, target_th, target_tf = get_utilizations(u_target)
                    dist = math.hypot(u_target.x - task.x, u_target.y - task.y)
                    e_add = task.energy_cost + ENERGY_PER_METER * dist
                    h_add = task.hover_time + dist / UAV_SPEED
                    f_add = task.compute_load
                    
                    if (target_te + e_add <= u_target.max_energy and
                        target_th + h_add <= u_target.max_hover_time and
                        (task.compute_load == 0 or target_tf + f_add <= u_target.max_compute)):
                        
                        score = -dist + 10.0 * (1.0 - (target_te + e_add) / u_target.max_energy)
                        if score > best_score:
                            best_score = score
                            best_target_uav = u_target
                            
                if best_target_uav is not None:
                    u_over.assigned_tasks.remove(task)
                    best_target_uav.assigned_tasks.append(task)
                    task.assigned_uav = best_target_uav
                    reallocated_any = True
                    print(f"      Migrated Task {task.task_id} from UAV {u_over.uav_id} to UAV {best_target_uav.uav_id}")
                    te, th, tf = get_utilizations(u_over)
                    if te <= u_over.max_energy and th <= u_over.max_hover_time and tf <= u_over.max_compute:
                        break
                        
        if not reallocated_any:
            print("      Cannot migrate tasks without violating target capacities. Evicting lowest priority tasks...")
            for overflow, u_over in overloaded_uavs:
                te, th, tf = get_utilizations(u_over)
                if te <= u_over.max_energy and th <= u_over.max_hover_time and tf <= u_over.max_compute:
                    continue
                u_over.assigned_tasks.sort(key=lambda t: -t.priority)
                while u_over.assigned_tasks and (te > u_over.max_energy or th > u_over.max_hover_time or tf > u_over.max_compute):
                    evicted = u_over.assigned_tasks.pop(0)
                    evicted.assigned_uav = None
                    print(f"      Evicted Task {evicted.task_id} (Priority {evicted.priority}) from UAV {u_over.uav_id} to prevent overload")
                    te, th, tf = get_utilizations(u_over)
            break

    for u in active_uavs:
        u.reset_resources()
        for t in u.assigned_tasks:
            u.consume_resources(t)

    return uavs


def som_assign(tasks, uavs, base_x=0.0, base_y=0.0, optimize=True):
    if not tasks or not uavs:
        return {}

    active_uavs = [u for u in uavs if u.active]
    if not active_uavs:
        return {t.task_id: None for t in tasks}

    uav_map = {u.uav_id: u for u in active_uavs}
    
    rem_energy = {u.uav_id: u.remaining_energy for u in active_uavs}
    rem_hover = {u.uav_id: u.remaining_hover_time for u in active_uavs}
    rem_compute = {u.uav_id: u.remaining_compute for u in active_uavs}

    uav_features = {u.uav_id: uav_feature(u) for u in active_uavs}

    learn_rate = SOM_LEARN_RATE
    lr_decay   = learn_rate / SOM_ITERATIONS

    for r in range(SOM_ITERATIONS):
        shuffled = list(tasks)
        np.random.default_rng(r).shuffle(shuffled)

        for task in shuffled:
            tf = task_feature(task)
            best_dist  = float('inf')
            best_uav   = None

            for uav in active_uavs:
                if optimize:
                    feat = uav_features[uav.uav_id]
                    uav_proxy = TempUAVWeightProxy(
                        feat, uav.uav_id,
                        rem_energy[uav.uav_id],
                        rem_hover[uav.uav_id],
                        rem_compute[uav.uav_id],
                        uav_map
                    )
                    d = matching_distance(task, uav_proxy, base_x, base_y)
                else:
                    # Baseline weight-bypass bug (original uav instead of proxy weights)
                    d = matching_distance(task, uav, base_x, base_y)
                if d < best_dist:
                    best_dist = d
                    best_uav  = uav

            if best_uav is None:
                continue

            for uav in active_uavs:
                n = neighbourhood(uav, best_uav, active_uavs)
                if n == 0.0:
                    continue
                feat   = uav_features[uav.uav_id]
                feat  += n * learn_rate * (tf - feat)
                uav_features[uav.uav_id] = feat

        learn_rate = max(0.01, learn_rate - lr_decay)

    assignment = {}
    sorted_tasks = sorted(tasks, key=lambda t: t.priority)

    for task in sorted_tasks:
        best_score = -float('inf')
        best_uav  = None

        for uav in active_uavs:
            if optimize:
                feat = uav_features[uav.uav_id]
                uav_proxy = TempUAVWeightProxy(
                    feat, uav.uav_id,
                    rem_energy[uav.uav_id],
                    rem_hover[uav.uav_id],
                    rem_compute[uav.uav_id],
                    uav_map
                )
                score = assignment_score_proxy(task, uav_proxy, base_x, base_y)
            else:
                # Baseline distance-only heuristic
                score = -math.hypot(uav.x - task.x, uav.y - task.y)
            if score > best_score:
                best_score = score
                best_uav  = uav

        if best_uav is not None:
            assignment[task.task_id] = best_uav
            dist = math.hypot(best_uav.x - task.x, best_uav.y - task.y)
            e_needed = task.energy_cost + ENERGY_PER_METER * dist
            h_needed = task.hover_time + dist / UAV_SPEED
            rem_energy[best_uav.uav_id] -= e_needed
            rem_hover[best_uav.uav_id] -= h_needed
            rem_compute[best_uav.uav_id] -= task.compute_load
        else:
            assignment[task.task_id] = None

    return assignment


# ----------------------------------------------------------
# PR-MODULE PUBLIC API
# ----------------------------------------------------------

def preassign(tasks, uavs, base_x=0.0, base_y=0.0, optimize=True):
    for uav in uavs:
        uav.clear_tasks()
        uav.reset_resources()

    assignment = som_assign(tasks, uavs, base_x, base_y, optimize)

    for task in tasks:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav

    if optimize:
        uavs = redistribute_load(uavs, base_x, base_y)
    _print_assignment_summary("PRE-ASSIGNMENT", uavs, tasks)
    return uavs


def reassign_new_tasks(new_tasks, uavs, base_x=0.0, base_y=0.0, optimize=True):
    assignment = som_assign(new_tasks, uavs, base_x, base_y, optimize)

    for task in new_tasks:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav

    if optimize:
        uavs = redistribute_load(uavs, base_x, base_y)
    print(f"\n[PR] Inserted {len(new_tasks)} new tasks.")
    return uavs


def reassign_after_location_update(updated_tasks, uavs,
                                   base_x=0.0, base_y=0.0, optimize=True):
    for task in updated_tasks:
        old_uav = task.assigned_uav
        if old_uav is not None and task in old_uav.assigned_tasks:
            old_uav.assigned_tasks.remove(task)
            old_uav.remaining_energy     += task.energy_cost
            old_uav.remaining_hover_time += task.hover_time
            old_uav.remaining_compute    += task.compute_load

    assignment = som_assign(updated_tasks, uavs, base_x, base_y, optimize)

    for task in updated_tasks:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav

    if optimize:
        uavs = redistribute_load(uavs, base_x, base_y)
    print(f"\n[PR] Re-assigned {len(updated_tasks)} location-updated tasks.")
    return uavs


def reassign_after_uav_failure(failed_uav, uavs,
                               base_x=0.0, base_y=0.0, optimize=True):
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
    assignment  = som_assign(orphaned, active_uavs, base_x, base_y, optimize)

    redistributed = 0
    for task in orphaned:
        uav = assignment.get(task.task_id)
        if uav is not None:
            uav.assigned_tasks.append(task)
            uav.consume_resources(task)
            task.assigned_uav = uav
            redistributed += 1

    if optimize:
        uavs = redistribute_load(uavs, base_x, base_y)
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
            f"  UAV {uav.uav_id:02d} (type {uav.uav_type:+d}) -> "
            f"{len(uav.assigned_tasks)} tasks  (hi-pri={hi})  "
            f"rem-compute={uav.remaining_compute:.1f}  "
            f"rem-hover={uav.remaining_hover_time:.0f}s"
        )
    print()