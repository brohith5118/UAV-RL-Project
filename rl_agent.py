import random
import numpy as np

from utils import calculate_reward

from config import (
    EPOCHS,
    RL_ALPHA,
    RL_GAMMA,
    EPSILON
)


class QLearningTrajectoryPlanner:

    def __init__(self, uav, tasks):

        self.uav = uav
        self.tasks = tasks

        self.num_tasks = len(tasks)

        if self.num_tasks == 0:
            raise ValueError(
                "No assigned tasks for trajectory planning"
            )

        self.q_table = np.zeros(
            (
                self.num_tasks,
                self.num_tasks
            )
        )

    def train(self):

        for epoch in range(EPOCHS):

            unvisited = list(
                range(self.num_tasks)
            )

            current_state = random.choice(
                unvisited
            )

            current_position = (
                self.tasks[current_state].x,
                self.tasks[current_state].y
            )

            unvisited.remove(current_state)

            while unvisited:

                # ε-greedy exploration

                if random.uniform(0, 1) < EPSILON:

                    next_state = random.choice(
                        unvisited
                    )

                else:

                    next_state = max(
                        unvisited,
                        key=lambda x:
                        self.q_table[
                            current_state,
                            x
                        ]
                    )

                next_task = self.tasks[next_state]

                reward = calculate_reward(
                    current_position,
                    next_task
                )

                future_q = max(
                    [
                        self.q_table[next_state, u]
                        for u in unvisited
                        if u != next_state
                    ],
                    default=0
                )

                self.q_table[
                    current_state,
                    next_state
                ] = (
                    (1 - RL_ALPHA)
                    * self.q_table[
                        current_state,
                        next_state
                    ]
                    +
                    RL_ALPHA
                    * (
                        reward
                        + RL_GAMMA * future_q
                    )
                )

                current_state = next_state

                current_position = (
                    next_task.x,
                    next_task.y
                )

                unvisited.remove(next_state)

    def get_best_route(self):

        # Start from nearest task to UAV

        start_idx = min(
            range(self.num_tasks),
            key=lambda i:
            (
                (self.tasks[i].x - self.uav.x) ** 2
                +
                (self.tasks[i].y - self.uav.y) ** 2
            )
        )

        route = [start_idx]

        unvisited = [
            i for i in range(self.num_tasks)
            if i != start_idx
        ]

        current = start_idx

        while unvisited:

            next_node = max(
                unvisited,
                key=lambda x:
                self.q_table[current, x]
            )

            route.append(next_node)

            unvisited.remove(next_node)

            current = next_node

        return [
            self.tasks[i]
            for i in route
        ]