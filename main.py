from environment import UAVEnvironment

from scheduler import assign_tasks
from rl_agent import QLearningScheduler
from visualization import plot_route


def main():

    print("Creating UAV environment...")

    env = UAVEnvironment()

    tasks, uavs = env.reset()

    print(f"Generated {len(tasks)} tasks")
    print(f"Generated {len(uavs)} UAVs")

    print("\nAssigning tasks to UAVs...")

    assign_tasks(tasks, uavs)

    print("\n--- Final Drone Assignments ---")

    for uav in uavs:

        print(
            f"UAV {uav.uav_id} "
            f"-> {len(uav.assigned_tasks)} tasks"
        )

    # Select UAV with most tasks
    target_uav = max(
        uavs,
        key=lambda u: len(u.assigned_tasks)
    )

    print(
        f"\nSelected UAV {target_uav.uav_id} "
        f"for route optimization"
    )

    if len(target_uav.assigned_tasks) == 0:

        print("No tasks assigned.")
        return

    print("\nTraining RL agent...")

    agent = QLearningScheduler(
        target_uav.assigned_tasks
    )

    agent.train()

    best_route = agent.get_best_route()

    print("Training complete!")

    plot_route(
        target_uav,
        target_uav.assigned_tasks,
        best_route
    )


if __name__ == "__main__":
    main()