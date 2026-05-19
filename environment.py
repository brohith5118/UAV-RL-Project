import random
from config import *
from uav import UAV
from task import Task


class Environment:
    def __init__(self):
        self.uavs = []
        self.tasks = []

    def generate_uavs(self):
        for i in range(NUM_UAVS):
            x = random.randint(0, MAP_SIZE)
            y = random.randint(0, MAP_SIZE)

            uav = UAV(
                uav_id=i,
                x=x,
                y=y,
                battery=MAX_BATTERY
            )

            self.uavs.append(uav)

    def generate_tasks(self):
        
        for i in range(NUM_TASKS):
            x = random.randint(0, MAP_SIZE)
            y = random.randint(0, MAP_SIZE)
            priority = random.randint(1, 5)

            task = Task(
                task_id=i,
                x=x,
                y=y,
                priority=priority
            )

            self.tasks.append(task)