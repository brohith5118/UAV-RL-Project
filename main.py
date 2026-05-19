from visualization import plot_environment
from environment import Environment
from scheduler import Scheduler


def main():

    env = Environment()

    env.generate_uavs()
    env.generate_tasks()
    scheduler = Scheduler(env)

    plot_environment(env)
    while len(env.tasks) != 0:

        scheduler.assign_tasks()

        print("\nTASK ASSIGNMENTS\n")

        for uav in env.uavs:
            print(f"UAV {uav.id}")
            print(f"Battery Left: {uav.battery:.2f}")

            task_ids = [task.id for task in uav.tasks]

            print(f"Assigned Tasks: {task_ids}")
            print()
        scheduler.complete_one_task()
        plot_environment(env)




if __name__ == "__main__":
    main()
