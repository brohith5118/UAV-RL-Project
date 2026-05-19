import matplotlib.pyplot as plt


def plot_environment(env):

    for uav in env.uavs:
        plt.scatter(uav.x, uav.y, marker='^', s=200)
        plt.text(uav.x, uav.y, f"UAV {uav.id}")

    for task in env.tasks:
        plt.scatter(task.x, task.y, marker='o')
        plt.text(task.x, task.y, f"T{task.id}")

    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.title("UAV Task Allocation")

    plt.show()