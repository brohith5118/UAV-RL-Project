# =========================================================
# PROPOSED METHOD VISUALIZATION
#
# Customizes plot titles and layouts to represent the new
# R-RL-AC framework.
# =========================================================

import sys
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add parent directory to path so we can import root modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import MAP_WIDTH, MAP_HEIGHT, UAV_SPEED
from utils  import estimate_finish_time, check_deadline
from visualization import (
    UAV_COLORS, PRIORITY_COLORS, PRIORITY_LABELS, _uav_color,
    plot_demand_map, plot_region_partition, plot_trajectories,
    plot_resource_utilisation, plot_priority_breakdown, plot_deadline_compliance,
    plot_demand_map_figure, plot_region_partition_figure
)


def plot_trajectories_figure(uavs, routes, demand_map, save_dir=None, prefix=""):
    """Standalone figure: Proposed Rollout RL UAV trajectories."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        'Proposed Rollout RL UAV Trajectories (TSA Module)\n'
        'Numbers indicate execution order',
        fontsize=13, fontweight='bold',
    )
    plot_trajectories(ax, uavs, routes, demand_map)
    plt.tight_layout()
    if save_dir:
        import os
        path = os.path.join(save_dir, f"{prefix}trajectories.png")
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_all(uavs, routes, tasks, demand_map, save_dir=None, prefix=""):
    """
    Proposed plot function with updated analytics dashboard title.
    """
    plot_demand_map_figure(demand_map, tasks, save_dir=save_dir, prefix=prefix)
    plot_region_partition_figure(uavs, tasks, demand_map, save_dir=save_dir, prefix=prefix)
    plot_trajectories_figure(uavs, routes, demand_map, save_dir=save_dir, prefix=prefix)

    # Figure 4: Analytics dashboard
    fig = plt.figure(figsize=(18, 10))
    active_uavs = [u for u in uavs if u.active]
    x_i = [len(u.assigned_tasks) for u in active_uavs]
    if x_i and sum(x_i) > 0:
        n_active = len(x_i)
        jains_index = (sum(x_i) ** 2) / (n_active * sum(val ** 2 for val in x_i))
    else:
        jains_index = 0.0

    fig.suptitle(
        f"Proposed R-RL-AC: Mission Analytics Dashboard (Jain's Fairness Index: {jains_index:.3f})",
        fontsize=13, fontweight='bold',
    )

    gs = fig.add_gridspec(2, 1, hspace=0.42, wspace=0.32)

    ax4 = fig.add_subplot(gs[0, 0])
    ax5 = fig.add_subplot(gs[1, 0])

    plot_resource_utilisation(ax4, uavs)
    plot_priority_breakdown(ax5, uavs)


    plt.tight_layout()
    if save_dir:
        import os
        path = os.path.join(save_dir, f"{prefix}analytics_dashboard.png")
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()

    fig_deadline, ax_deadline = plt.subplots(figsize=(8, 5))

    fig_deadline.suptitle(
        "Mission Deadline Compliance Analysis",
        fontsize=13,
        fontweight='bold'
    )

    plot_deadline_compliance(
        ax_deadline,
        uavs,
        routes
    )

    fig_deadline.tight_layout(rect=[0, 0, 1, 0.93])

    if save_dir:
        import os
        path = os.path.join(
            save_dir,
            f"{prefix}deadline_compliance.png"
        )
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_reward_convergence(reward_logs: dict, save_dir=None, prefix=""):
    """
    Proposed convergence plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for uid, rewards in reward_logs.items():
        if not rewards:
            continue
        arr     = np.array(rewards, dtype=float)
        kernel  = min(20, len(arr))
        smooth  = np.convolve(
            arr, np.ones(kernel) / kernel, mode='valid'
        )
        ax.plot(smooth, label=f'UAV {uid}',
                color=_uav_color(uid), linewidth=1.2)

    ax.set_xlabel('Episode / Step')
    ax.set_ylabel('Route Reward')
    ax.set_title('Proposed TSA Rollout RL Convergence (Instantaneous Optimal Policy)')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_dir:
        import os
        path = os.path.join(save_dir, f"{prefix}convergence.png")
        plt.savefig(path, dpi=150)
        plt.close()
    else:
        plt.show()
