import random

from task import Task
from uav import UAV

from config import (
    MAP_WIDTH,
    MAP_HEIGHT,
    NUM_UAVS,
    MIN_ENERGY,
    MAX_ENERGY,
    MIN_HOVER_TIME,
    MAX_HOVER_TIME,
    MIN_COMPUTE,
    MAX_COMPUTE
)

def generate_tasks(
    num_tasks,
    high_priority_ratio
):

    task_list = []

    for task_id in range(num_tasks):

        x = random.randint(0, MAP_WIDTH - 1)
        y = random.randint(0, MAP_HEIGHT - 1)

        if random.random() < high_priority_ratio:
            priority = 1
        else:
            priority = 2

        energy_cost = random.uniform(5, 20)
        hover_time = random.uniform(3, 10)
        compute_load = random.uniform(10, 30)

        task = Task(
            task_id,
            x,
            y,
            priority,
            energy_cost,
            hover_time,
            compute_load
        )

        task_list.append(task)

    return task_list


def generate_uavs():

    uavs = []

    for uav_id in range(NUM_UAVS):

        x = random.randint(0, MAP_WIDTH - 1)
        y = random.randint(0, MAP_HEIGHT - 1)

        max_energy = random.uniform(
            MIN_ENERGY,
            MAX_ENERGY
        )

        max_hover = random.uniform(
            MIN_HOVER_TIME,
            MAX_HOVER_TIME
        )

        max_compute = random.uniform(
            MIN_COMPUTE,
            MAX_COMPUTE
        )

        uav = UAV(
            uav_id,
            x,
            y,
            max_energy,
            max_hover,
            max_compute
        )

        uavs.append(uav)

    return uavs