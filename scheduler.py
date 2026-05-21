import math

from config import (
    ALPHA,
    GAMMA,
    RHO,
    ITERATIONS,
    LAMBDA_TV
)


# =========================================================
# CAPACITY FEASIBILITY CHECK
#
# Enforces Equation (5):
#
# Σ ω_k,i x_ui <= C_u,k
#
# =========================================================

def is_feasible(uav, task):

    current_energy = sum(
        t.energy_cost
        for t in uav.assigned_tasks
    )

    current_hover = sum(
        t.hover_time
        for t in uav.assigned_tasks
    )

    current_compute = sum(
        t.compute_load
        for t in uav.assigned_tasks
    )

    energy_ok = (
        current_energy
        + task.energy_cost
        <= uav.max_energy
    )

    hover_ok = (
        current_hover
        + task.hover_time
        <= uav.max_hover_time
    )

    compute_ok = (
        current_compute
        + task.compute_load
        <= uav.max_compute
    )

    return (
        energy_ok
        and hover_ok
        and compute_ok
    )


# =========================================================
# GENERALIZED COST FUNCTION
#
# Equation (7):
#
# δ_ui =
# α d(p_u(t), g_i)
# - γ w_pri_i
# + Σ μ_u,k ω_k,i
#
# =========================================================

def generalized_cost(uav, task):

    # ---------------------------------------------
    # DISTANCE TERM
    # ---------------------------------------------

    distance = math.hypot(
        uav.x - task.x,
        uav.y - task.y
    )

    # ---------------------------------------------
    # PRIORITY TERM
    #
    # Higher priority should reduce cost
    # ---------------------------------------------

    if task.priority == 1:
        priority_weight = 100
    else:
        priority_weight = 20

    # ---------------------------------------------
    # LAGRANGE MULTIPLIER PENALTIES
    #
    # μ_u,k ω_k,i
    # ---------------------------------------------

    lagrange_penalty = (

        uav.mu_energy
        * task.energy_cost

        +

        uav.mu_hover
        * task.hover_time

        +

        uav.mu_compute
        * task.compute_load
    )

    # ---------------------------------------------
    # FINAL GENERALIZED COST
    # ---------------------------------------------

    total_cost = (

        ALPHA * distance

        -

        GAMMA * priority_weight

        +

        lagrange_penalty
    )

    return total_cost


# =========================================================
# TV REGULARIZATION APPROXIMATION
#
# Encourages spatially compact regions
#
# λTV Σ |x_ui - x_uj|
#
# Approximated using distance to region centroid
# =========================================================

def compactness_penalty(uav, task):

    # First task has no penalty

    if len(uav.assigned_tasks) == 0:
        return 0

    centroid_x = sum(
        t.x for t in uav.assigned_tasks
    ) / len(uav.assigned_tasks)

    centroid_y = sum(
        t.y for t in uav.assigned_tasks
    ) / len(uav.assigned_tasks)

    centroid_distance = math.hypot(
        task.x - centroid_x,
        task.y - centroid_y
    )

    return (
        LAMBDA_TV
        * centroid_distance
    )


# =========================================================
# LAGRANGE MULTIPLIER UPDATE
#
# Equation (8):
#
# μ_u,k ←
# [ μ_u,k + ρ(usage - capacity) ]+
#
# =========================================================

def update_lagrange_multipliers(uav):

    total_energy = sum(
        t.energy_cost
        for t in uav.assigned_tasks
    )

    total_hover = sum(
        t.hover_time
        for t in uav.assigned_tasks
    )

    total_compute = sum(
        t.compute_load
        for t in uav.assigned_tasks
    )

    # ---------------------------------------------
    # ENERGY MULTIPLIER
    # ---------------------------------------------

    uav.mu_energy = max(

        0.0,

        uav.mu_energy
        +
        RHO * (
            total_energy
            - uav.max_energy
        )
    )

    # ---------------------------------------------
    # HOVER MULTIPLIER
    # ---------------------------------------------

    uav.mu_hover = max(

        0.0,

        uav.mu_hover
        +
        RHO * (
            total_hover
            - uav.max_hover_time
        )
    )

    # ---------------------------------------------
    # COMPUTE MULTIPLIER
    # ---------------------------------------------

    uav.mu_compute = max(

        0.0,

        uav.mu_compute
        +
        RHO * (
            total_compute
            - uav.max_compute
        )
    )


# =========================================================
# CAPACITY-CONSTRAINED REGION PARTITIONING
#
# Solves:
#
# min Σ δ_ui x_ui
# + λTV compactness
#
# subject to:
#
# Σ ω_k,i x_ui <= C_u,k
#
# =========================================================

def assign_tasks(task_list, uavs):

    # =====================================================
    # INITIALIZATION
    # =====================================================

    for uav in uavs:

        uav.clear_tasks()

        uav.mu_energy = 0.0
        uav.mu_hover = 0.0
        uav.mu_compute = 0.0

    # =====================================================
    # ITERATIVE POWER-DIAGRAM OPTIMIZATION
    # =====================================================

    for iteration in range(ITERATIONS):

        # Clear assignments for reassignment

        for uav in uavs:
            uav.clear_tasks()

        # -------------------------------------------------
        # SORT TASKS BY PRIORITY
        #
        # High-priority tasks allocated first
        # -------------------------------------------------

        sorted_tasks = sorted(

            task_list,

            key=lambda t: (
                t.priority,
                -(t.energy_cost
                  + t.hover_time
                  + t.compute_load)
            )
        )

        # =================================================
        # TASK ASSIGNMENT
        # =================================================

        unassigned_tasks = []

        for task in sorted_tasks:

            feasible_candidates = []

            # ---------------------------------------------
            # FIND FEASIBLE UAVS
            # ---------------------------------------------

            for uav in uavs:

                if not is_feasible(
                    uav,
                    task
                ):
                    continue

                # -----------------------------------------
                # Compute generalized cost
                # -----------------------------------------

                cost = generalized_cost(
                    uav,
                    task
                )

                # -----------------------------------------
                # TV regularization
                # -----------------------------------------

                cost += compactness_penalty(
                    uav,
                    task
                )

                feasible_candidates.append(
                    (cost, uav)
                )

            # ---------------------------------------------
            # ASSIGN TO MINIMUM COST UAV
            # ---------------------------------------------

            if feasible_candidates:

                feasible_candidates.sort(
                    key=lambda x: x[0]
                )

                selected_uav = feasible_candidates[0][1]

                selected_uav.assigned_tasks.append(
                    task
                )

            else:

                unassigned_tasks.append(task)

        # =================================================
        # UPDATE LAGRANGE MULTIPLIERS
        # =================================================

        for uav in uavs:

            update_lagrange_multipliers(
                uav
            )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print("\n=== PARTITIONING SUMMARY ===")

    for uav in uavs:

        total_energy = sum(
            t.energy_cost
            for t in uav.assigned_tasks
        )

        total_hover = sum(
            t.hover_time
            for t in uav.assigned_tasks
        )

        total_compute = sum(
            t.compute_load
            for t in uav.assigned_tasks
        )

        print(
            f"\nUAV {uav.uav_id}"
        )

        print(
            f"Tasks: {len(uav.assigned_tasks)}"
        )

        print(
            f"Energy: "
            f"{total_energy:.2f}/"
            f"{uav.max_energy:.2f}"
        )

        print(
            f"Hover: "
            f"{total_hover:.2f}/"
            f"{uav.max_hover_time:.2f}"
        )

        print(
            f"Compute: "
            f"{total_compute:.2f}/"
            f"{uav.max_compute:.2f}"
        )

    return uavs