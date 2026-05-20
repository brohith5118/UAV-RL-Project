import matplotlib.pyplot as plt


def plot_route(target_uav, tasks, best_route):

    route_x = [target_uav.x] + [t.x for t in best_route]
    route_y = [target_uav.y] + [t.y for t in best_route]

    plt.figure(figsize=(10, 6))

    plt.plot(
        route_x,
        route_y,
        marker='o',
        linestyle='-',
        color='b',
        alpha=0.7,
        label='Flight Path'
    )

    plt.plot(
        target_uav.x,
        target_uav.y,
        marker='s',
        color='green',
        markersize=12,
        label='UAV Base'
    )

    for t in tasks:

        if t.priority == 1:

            plt.scatter(
                t.x,
                t.y,
                color='red',
                s=100,
                zorder=5,
                label='High Priority'
                if 'High Priority'
                not in plt.gca().get_legend_handles_labels()[1]
                else ""
            )

        else:

            plt.scatter(
                t.x,
                t.y,
                color='orange',
                s=50,
                zorder=5,
                label='Normal Priority'
                if 'Normal Priority'
                not in plt.gca().get_legend_handles_labels()[1]
                else ""
            )

    plt.title("Optimized UAV Flight Sequence")

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    plt.legend()

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()