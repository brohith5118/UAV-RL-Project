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
import pandas as pd

# =========================================================
# CREATE OUTPUT FOLDER AUTOMATICALLY
# =========================================================
OUTPUT_FOLDER = "generated_graphs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Add parent directory to path so we can import root modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config import MAP_WIDTH, MAP_HEIGHT, UAV_SPEED, GRID_RESOLUTION, ENERGY_PER_METER
from utils  import estimate_finish_time, check_deadline
from visualization import (
    UAV_COLORS, PRIORITY_COLORS, PRIORITY_LABELS, _uav_color,
    plot_demand_map, plot_region_partition, plot_trajectories,
    plot_resource_utilisation, plot_priority_breakdown, plot_deadline_compliance,
    plot_demand_map_figure, plot_region_partition_figure
)


def get_save_path(filename):
    """
    Returns full save path inside output folder.
    """
    return os.path.join(OUTPUT_FOLDER, filename)


def plot_trajectories_figure(uavs, routes, demand_map, prefix=""):
    """Standalone figure: Proposed Rollout RL UAV trajectories."""
    
    fig, ax = plt.subplots(figsize=(8, 7))

    fig.suptitle(
        'Proposed Rollout RL UAV Trajectories (TSA Module)\n'
        'Numbers indicate execution order',
        fontsize=13,
        fontweight='bold',
    )

    plot_trajectories(ax, uavs, routes, demand_map)

    plt.tight_layout()

    path = get_save_path(f"{prefix}trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {path}")


def plot_all(uavs, routes, tasks, demand_map, prefix=""):
    """
    Proposed plot function with updated analytics dashboard title.
    """

    # Existing plots from visualization.py
    plot_demand_map_figure(
        demand_map,
        tasks,
        save_dir=OUTPUT_FOLDER,
        prefix=prefix
    )

    plot_region_partition_figure(
        uavs,
        tasks,
        demand_map,
        save_dir=OUTPUT_FOLDER,
        prefix=prefix
    )

    plot_trajectories_figure(
        uavs,
        routes,
        demand_map,
        prefix=prefix
    )

    plot_task_details(
        tasks,
        prefix=prefix
    )

    # =====================================================
    # Figure 4: Analytics Dashboard
    # =====================================================

    fig = plt.figure(figsize=(18, 10))

    active_uavs = [u for u in uavs if u.active]

    x_i = [len(u.assigned_tasks) for u in active_uavs]

    if x_i and sum(x_i) > 0:
        n_active = len(x_i)

        jains_index = (
            (sum(x_i) ** 2)
            / (n_active * sum(val ** 2 for val in x_i))
        )

    else:
        jains_index = 0.0

    fig.suptitle(
        f"Proposed R-RL-AC: Mission Analytics Dashboard "
        f"(Jain's Fairness Index: {jains_index:.3f})",
        fontsize=13,
        fontweight='bold',
    )

    gs = fig.add_gridspec(
        2,
        1,
        hspace=0.42,
        wspace=0.32
    )

    ax4 = fig.add_subplot(gs[0, 0])
    ax5 = fig.add_subplot(gs[1, 0])

    plot_resource_utilisation(ax4, uavs)
    plot_priority_breakdown(ax5, uavs)

    plt.tight_layout()

    path = get_save_path(f"{prefix}analytics_dashboard.png")

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {path}")

    # =====================================================
    # Deadline Compliance Figure
    # =====================================================

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

    path = get_save_path(
        f"{prefix}deadline_compliance.png"
    )

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {path}")


def plot_reward_convergence(reward_logs: dict, prefix=""):
    """
    Proposed convergence plot.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    for uid, rewards in reward_logs.items():

        if not rewards:
            continue

        arr = np.array(rewards, dtype=float)

        kernel = min(20, len(arr))

        smooth = np.convolve(
            arr,
            np.ones(kernel) / kernel,
            mode='valid'
        )

        ax.plot(
            smooth,
            label=f'UAV {uid}',
            color=_uav_color(uid),
            linewidth=1.2
        )

    ax.set_xlabel('Episode / Step')
    ax.set_ylabel('Route Reward')

    ax.set_title(
        'Proposed TSA Rollout RL Convergence '
        '(Instantaneous Optimal Policy)'
    )

    ax.legend(fontsize=7, ncol=3)

    ax.grid(
        linestyle='--',
        alpha=0.5
    )

    plt.tight_layout()

    path = get_save_path(f"{prefix}convergence.png")

    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[SAVED] {path}")

def plot_task_details(tasks, prefix=""):
    """
    Creates a task information image for presentation/report.
    """

    rows = []

    for t in tasks:

        rows.append([
            t.task_id,
            round(t.x, 2),
            round(t.y, 2),
            t.priority,
            round(t.deadline, 2),
            getattr(t, "task_type", "N/A"),
            round(getattr(t, "energy_cost", 0), 2),
            round(getattr(t, "hover_time", 0), 2),
            round(getattr(t, "compute_load", 0), 2)
        ])

    columns = [
        "Task ID",
        "X",
        "Y",
        "Priority",
        "Deadline",
        "Type",
        "Energy",
        "Hover",
        "Compute",
    ]

    df = pd.DataFrame(rows, columns=columns)

    fig_height = max(6, len(df) * 0.35)

    fig, ax = plt.subplots(
        figsize=(16, fig_height)
    )

    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)

    plt.title(
        "Task Details Summary",
        fontsize=16,
        fontweight="bold",
        pad=20
    )

    path = get_save_path(
        f"{prefix}task_details.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[SAVED] {path}")