import random

from task import Task
from uav import UAV

from config import (
    MAP_WIDTH,
    MAP_HEIGHT,

    NUM_TASKS,
    HIGH_PRIORITY_RATIO,

    NUM_UAVS,

    MIN_ENERGY,
    MAX_ENERGY,

    MIN_COMPUTE,
    MAX_COMPUTE
)


class UAVEnvironment:

    def __init__(self):

        self.tasks = []
        self.uavs = []

    # ======================================
    # RANDOM TASK GENERATION
    # ======================================

    def generate_tasks(self):

        self.tasks = []

        for task_id in range(NUM_TASKS):

            x = random.randint(0, MAP_WIDTH - 1)
            y = random.randint(0, MAP_HEIGHT - 1)

            if random.random() < HIGH_PRIORITY_RATIO:
                priority = 1
                workload = 15.0
            else:
                priority = 2
                workload = 10.0

            task = Task(
                task_id,
                x,
                y,
                priority,
                workload
            )

            self.tasks.append(task)

    # ======================================
    # RANDOM UAV GENERATION
    # ======================================

    def generate_uavs(self):

        self.uavs = []

        for uav_id in range(NUM_UAVS):

            x = random.randint(0, MAP_WIDTH - 1)
            y = random.randint(0, MAP_HEIGHT - 1)

            max_energy = random.randint(
                MIN_ENERGY,
                MAX_ENERGY
            )

            max_compute = random.randint(
                MIN_COMPUTE,
                MAX_COMPUTE
            )

            uav = UAV(
                uav_id=uav_id,
                x=x,
                y=y,
                max_energy=max_energy,
                max_compute=max_compute
            )

            self.uavs.append(uav)

    # ======================================
    # INITIALIZE FULL ENVIRONMENT
    # ======================================

    def reset(self):

        self.generate_tasks()
        self.generate_uavs()

        return self.tasks, self.uavs