from utils import euclidean_distance
from config import ENERGY_PER_DISTANCE


class Scheduler:
    def __init__(self, env):
        self.env = env
        self.completed_tasks = []
    

    def assign_tasks(self):
        for uav in self.env.uavs:
            uav.tasks = []

        for task in self.env.tasks:

            best_uav = None
            best_distance = float('inf')

            for uav in self.env.uavs:
                dist = euclidean_distance(
                    uav.position(),
                    task.position()
                )

                if dist < best_distance:
                    best_distance = dist
                    best_uav = uav

            energy_cost = best_distance * ENERGY_PER_DISTANCE

            if best_uav.battery > energy_cost:
                best_uav.assign_task(task)

    def complete_one_task(self):
        for uav in self.env.uavs:
            if len(uav.tasks) != 0:
                task = uav.tasks[0]
                
                dist = euclidean_distance(
                    uav.position(),
                    task.position()
                )

                energy_cost = dist * ENERGY_PER_DISTANCE

                uav.move_to(
                    task.x,
                    task.y,
                    energy_cost
                )

                task.completed = True
                print(f"UAV {uav.id} completed task {task.id}")

                if task in self.env.tasks:
                    self.env.tasks.remove(task)
                uav.tasks.remove(task)
                self.completed_tasks.append(task)

