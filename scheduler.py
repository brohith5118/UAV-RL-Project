import math

from config import ALPHA, GAMMA, RHO, ITERATIONS


def assign_tasks(task_list, uavs):

    for step in range(ITERATIONS):

        for uav in uavs:
            uav.clear_tasks()

        for task in task_list:

            best_cost = float('inf')
            best_uav = None

            for uav in uavs:

                distance = math.hypot(
                    uav.x - task.x,
                    uav.y - task.y
                )

                penalty = (
                    uav.penalty_energy * task.compute_workload
                    +
                    uav.penalty_compute * task.compute_workload
                )
                
                load_factor = len(uav.assigned_tasks) * 2

                cost = (
                    (ALPHA * distance)
                    -
                    (GAMMA * task.priority)
                    +
                    penalty
                    +
                    load_factor
                )

                if cost < best_cost:
                    best_cost = cost
                    best_uav = uav

            best_uav.assigned_tasks.append(task)

        for uav in uavs:

            total_energy_used = sum(
                t.compute_workload
                for t in uav.assigned_tasks
            )

            total_compute_used = sum(
                t.compute_workload
                for t in uav.assigned_tasks
            )

            uav.penalty_energy = max(
                0.0,
                uav.penalty_energy
                + RHO * (total_energy_used - uav.max_energy)
            )

            uav.penalty_compute = max(
                0.0,
                uav.penalty_compute
                + RHO * (total_compute_used - uav.max_compute)
            )