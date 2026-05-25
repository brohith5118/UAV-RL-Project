import random
import math
import numpy as np

from task import Task
from uav  import UAV

from config import (
    MAP_WIDTH,
    MAP_HEIGHT,
    GRID_RESOLUTION,
    NUM_UAVS,
    NUM_TASKS,
    HIGH_PRIORITY_RATIO,
    PRIORITY_DEADLINES,
    TASK_TYPE_RATIO,
    MIN_ENERGY,
    MAX_ENERGY,
    MIN_HOVER_TIME,
    MAX_HOVER_TIME,
    ENERGY_PER_METER,
    UAV_SPEED,
    UAV_TYPE_MAX_FLIGHT,
    UAV_TYPE_MAX_COMPUTE,
)


# ----------------------------------------------------------
# SYNTHETIC SENSING-DEMAND MAP
# ----------------------------------------------------------

def generate_demand_map(seed=102):
    rng = np.random.default_rng(seed)

    grid = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)

    ys = np.arange(MAP_HEIGHT)
    xs = np.arange(MAP_WIDTH)

    X, Y = np.meshgrid(xs, ys)

    # More variation than before
    num_hotspots = rng.integers(1, 16)

    hotspot_centers = []

    min_distance = min(MAP_WIDTH, MAP_HEIGHT) * 0.15

    for i in range(num_hotspots):

        for _attempt in range(50):

            cx = rng.uniform(0, MAP_WIDTH)
            cy = rng.uniform(0, MAP_HEIGHT)

            valid = True

            for px, py in hotspot_centers:

                distance = np.sqrt(
                    (cx - px) ** 2 +
                    (cy - py) ** 2
                )

                if distance < min_distance:
                    valid = False
                    break

            if valid:
                hotspot_centers.append((cx, cy))
                break

        hotspot_type = rng.choice(
            ["circular", "elliptical"],
            p=[0.6, 0.4]
        )

        weight = rng.uniform(0.3, 1.5)

        if hotspot_type == "circular":

            sigma = rng.uniform(2, 25)

            gauss = weight * np.exp(
                -(
                    (X - cx) ** 2 +
                    (Y - cy) ** 2
                )
                /
                (2 * sigma ** 2)
            )

        else:

            sigma_x = rng.uniform(2, 25)
            sigma_y = rng.uniform(2, 25)

            gauss = weight * np.exp(
                -(
                    ((X - cx) ** 2) / (2 * sigma_x ** 2)
                    +
                    ((Y - cy) ** 2) / (2 * sigma_y ** 2)
                )
            )

        grid += gauss

    if grid.max() > 0:

        relative_grid = grid / grid.max()

        noise = rng.uniform(
            0,
            0.2,
            size=grid.shape
        )

        grid += noise * (0.3 + relative_grid)

    grid = grid / (grid.max() + 1e-9)

    return grid.astype(np.float32)

def score_to_priority(score, high_ratio=HIGH_PRIORITY_RATIO):
    """
    Convert a continuous demand score to a discrete priority
    level used throughout the paper.
      score >= 1 - high_ratio  →  1 (critical)
      score >= 0.4             →  2 (important)
      else                     →  3 (routine)
    """
    if score >= (1.0 - high_ratio):
        return 1
    elif score >= 0.5:
        return 2
    else:
        return 3


def score_to_task_type(score,rng):
    """
    Task types:
        -1 : acquisition-only
         0 : balanced
         1 : compute-intensive

    Distribution controlled by TASK_TYPE_RATIO.
    """

    r = rng.random()

    if r < TASK_TYPE_RATIO[-1]:
        return -1

    elif r < TASK_TYPE_RATIO[-1] + TASK_TYPE_RATIO[0]:
        return 0

    else:
        return 1

# ----------------------------------------------------------
# TASK GENERATION
# ----------------------------------------------------------

def generate_tasks(
    num_tasks=NUM_TASKS,
    high_priority_ratio=HIGH_PRIORITY_RATIO,
    demand_map=None,
    seed=0
):
    rng = np.random.default_rng(seed)

    if demand_map is None:
        demand_map = generate_demand_map(seed=seed)

    # --------------------------------------------------
    # Sample task locations from demand map
    # --------------------------------------------------

    flat_demand = demand_map.flatten()
    probs = flat_demand / flat_demand.sum()

    cell_count = MAP_WIDTH * MAP_HEIGHT

    chosen_cells = rng.choice(
        cell_count,
        size=num_tasks,
        replace=False,
        p=probs
    )

    task_list = []

    for tid, cell_idx in enumerate(chosen_cells):

        row = cell_idx // MAP_WIDTH
        col = cell_idx % MAP_WIDTH

        # Cell-center coordinates
        x = col * GRID_RESOLUTION + 0.5
        y = row * GRID_RESOLUTION + 0.5
        # x = 0
        # y = 0

        score = rng.random()

        priority = score_to_priority(score, high_ratio=high_priority_ratio)

        task_type = score_to_task_type(score, rng)
        # task_type = 0

        # --------------------------------------------------
        # Workload generation
        # --------------------------------------------------

        energy_cost = float(rng.uniform(5, 20))

        hover_time = float(rng.uniform(3, 10))

        if task_type == 1:

            compute_load = float(
                rng.uniform(15, 30)
            )

        elif task_type == 0:

            compute_load = float(
                rng.uniform(5, 15)
            )

        else:

            compute_load = 0.0

        deadline = PRIORITY_DEADLINES[priority]

        task = Task(
            task_id=tid,
            x=x,
            y=y,
            priority=priority,
            task_type=task_type,
            energy_cost=energy_cost,
            hover_time=hover_time,
            compute_load=compute_load,
            deadline=deadline,
        )

        task_list.append(task)

    return task_list, demand_map

# ----------------------------------------------------------
# UAV GENERATION  (heterogeneous fleet)
# ----------------------------------------------------------

def generate_uavs(num_uavs=NUM_UAVS, seed=99):
    rng  = np.random.default_rng(seed)
    uavs = []

    type_pool = (
        [-1] * (num_uavs // 3) +
        [ 0] * (num_uavs // 3) +
        [ 1] * (num_uavs - 2 * (num_uavs // 3))
    )
    rng.shuffle(type_pool)

    for uid in range(num_uavs):

        x = float(rng.uniform(0, MAP_WIDTH))
        y = float(rng.uniform(0, MAP_HEIGHT))
        # x = 0
        # y = 0

        uav_type = type_pool[uid]
        # uav_type = 0

        # Max flight time from Table 1
        max_hover = float(UAV_TYPE_MAX_FLIGHT[uav_type])

        # Max energy  ∝  flight time (longer-endurance UAVs
        # carry more battery)
        max_energy = (
            max_hover * ENERGY_PER_METER * UAV_SPEED * 0.6
        )
        max_energy = float(np.clip(max_energy, MIN_ENERGY, MAX_ENERGY * 2))

        # Max compute from Table 1
        max_compute = float(UAV_TYPE_MAX_COMPUTE[uav_type])

        uav = UAV(
            uav_id         = uid,
            x              = x,
            y              = y,
            uav_type       = uav_type,
            max_energy     = max_energy,
            max_hover_time = max_hover,
            max_compute    = max_compute,
        )
        uavs.append(uav)

    return uavs


# ----------------------------------------------------------
# DYNAMIC EVENT HELPERS
# ----------------------------------------------------------

def generate_new_task(task_id, demand_map, seed=None):
    """Spawn a single new urgent task (paper Section 4.2)."""
    rng = np.random.default_rng(seed)

    x = float(rng.uniform(0, MAP_WIDTH))
    y = float(rng.uniform(0, MAP_HEIGHT))

    row = min(int(y), MAP_HEIGHT - 1)
    col = min(int(x), MAP_WIDTH  - 1)
    score = float(demand_map[row, col])

    priority  = 1 if rng.random() < 0.5 else 2  # new tasks tend urgent
    task_type = score_to_task_type(score, rng)

    deadlines = {1: 1000, 2: 1250, 3: 1500}  # Table 11

    return Task(
        task_id     = task_id,
        x           = x,
        y           = y,
        priority    = priority,
        task_type   = task_type,
        energy_cost = float(rng.uniform(5, 20)),
        hover_time  = float(rng.uniform(3, 10)),
        compute_load= float(rng.uniform(5, 15)) if task_type >= 0 else 0.0,
        deadline    = deadlines[priority],
    )