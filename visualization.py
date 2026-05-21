import matplotlib.pyplot as plt


def plot_all_routes(uavs, routes):

    plt.figure(figsize=(12, 10))

    # ---------------------------------------------------
    # Different colors for each UAV
    # ---------------------------------------------------

    colors = [
        'blue',
        'green',
        'purple',
        'cyan',
        'magenta',
        'brown',
        'black',
        'yellow',
        'pink',
        'gray'
    ]

    # ---------------------------------------------------
    # Plot each UAV region and trajectory
    # ---------------------------------------------------

    for idx, uav in enumerate(uavs):

        color = colors[idx % len(colors)]

        tasks = uav.assigned_tasks

        # -----------------------------------------------
        # Plot assigned tasks
        # -----------------------------------------------

        for task in tasks:

            if task.priority == 1:

                plt.scatter(
                    task.x,
                    task.y,
                    color='red',
                    s=100,
                    edgecolors='black',
                    zorder=5,
                    label='High Priority'
                    if (
                        idx == 0 and
                        'High Priority'
                        not in plt.gca().get_legend_handles_labels()[1]
                    )
                    else ""
                )

            else:

                plt.scatter(
                    task.x,
                    task.y,
                    color='orange',
                    s=50,
                    edgecolors='black',
                    zorder=5,
                    label='Normal Priority'
                    if (
                        idx == 0 and
                        'Normal Priority'
                        not in plt.gca().get_legend_handles_labels()[1]
                    )
                    else ""
                )

        # -----------------------------------------------
        # Plot UAV base
        # -----------------------------------------------

        plt.scatter(
            uav.x,
            uav.y,
            color=color,
            marker='s',
            s=250,
            edgecolors='black',
            label=f'UAV {uav.uav_id} Base'
        )

        # -----------------------------------------------
        # Plot trajectory
        # -----------------------------------------------

        route = routes[uav.uav_id]

        if len(route) == 0:
            continue

        route_x = [uav.x]
        route_y = [uav.y]

        for task in route:

            route_x.append(task.x)
            route_y.append(task.y)

        # Return to base

        route_x.append(uav.x)
        route_y.append(uav.y)

        plt.plot(
            route_x,
            route_y,
            linestyle='--',
            marker='o',
            linewidth=2,
            alpha=0.8,
            color=color,
            label=f'UAV {uav.uav_id} Trajectory'
        )

    # ---------------------------------------------------
    # Plot styling
    # ---------------------------------------------------

    plt.title(
        'Multi-UAV Capacity-Constrained Task Planning',
        fontsize=16
    )

    plt.xlabel(
        'X Coordinate',
        fontsize=12
    )

    plt.ylabel(
        'Y Coordinate',
        fontsize=12
    )

    plt.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    plt.show()