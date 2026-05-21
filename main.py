from environment import (
    generate_tasks,
    generate_uavs
)

from scheduler import assign_tasks

from rl_agent import (
    QLearningTrajectoryPlanner
)

from visualization import (
    plot_all_routes
)

from config import (
    NUM_TASKS,
    HIGH_PRIORITY_RATIO
)


def main():

    # GENERATE TASKS

    print(
        "Generating remote sensing tasks..."
    )

    tasks = generate_tasks(
        NUM_TASKS,
        HIGH_PRIORITY_RATIO
    )

    print(
        f"Generated {len(tasks)} tasks"
    )

    # GENERATE UAVS

    print(
        "\nGenerating heterogeneous UAV fleet..."
    )

    uavs = generate_uavs()

    print(
        f"Generated {len(uavs)} UAVs"
    )

    # CAPACITY-CONSTRAINED PARTITIONING

    print(
        "\nRunning region partitioning..."
    )

    assign_tasks(tasks, uavs)

    # DISPLAY FINAL PARTITIONS

    print(
        "\n=== FINAL REGION PARTITIONS ==="
    )

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
            f"Assigned Tasks: "
            f"{len(uav.assigned_tasks)}"
        )

        print(
            f"Energy Usage: "
            f"{total_energy:.2f}/"
            f"{uav.max_energy:.2f}"
        )

        print(
            f"Hover Usage: "
            f"{total_hover:.2f}/"
            f"{uav.max_hover_time:.2f}"
        )

        print(
            f"Compute Usage: "
            f"{total_compute:.2f}/"
            f"{uav.max_compute:.2f}"
        )

    # RL TRAJECTORY OPTIMIZATION

    print(
        "\nRunning RL trajectory optimization..."
    )

    all_routes = {}

    for uav in uavs:

        if len(uav.assigned_tasks) == 0:

            print(
                f"UAV {uav.uav_id} "
                f"has no assigned tasks."
            )

            all_routes[uav.uav_id] = []

            continue

        print(
            f"Training RL planner for "
            f"UAV {uav.uav_id}..."
        )

        planner = QLearningTrajectoryPlanner(
            uav,
            uav.assigned_tasks
        )

        planner.train()

        best_route = planner.get_best_route()

        all_routes[uav.uav_id] = best_route

    print(
        "\nTrajectory optimization complete."
    )


    plot_all_routes(
        uavs,
        all_routes
    )


if __name__ == '__main__':
    main()