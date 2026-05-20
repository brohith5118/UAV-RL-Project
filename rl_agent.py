import random
import numpy as np

from utils import calculate_reward
from config import (
    EPOCHS,
    RL_ALPHA,
    RL_GAMMA,
    EPSILON
)


class QLearningScheduler:

    def __init__(self, tasks):
        if len(tasks) == 0:
            raise ValueError(
                "No tasks assigned to this UAV."
            )
        
        self.tasks = tasks
        self.num_tasks = len(tasks)

        self.q_table = np.zeros(
            (self.num_tasks, self.num_tasks)
        )

    def train(self):

        for epoch in range(EPOCHS):

            unvisited = list(range(self.num_tasks))

            current_state = random.choice(unvisited)

            unvisited.remove(current_state)

            while unvisited:

                # Exploration
                if random.uniform(0, 1) < EPSILON:
                    next_state = random.choice(unvisited)

                # Exploitation
                else:
                    best_q = -float('inf')
                    best_next = unvisited[0]

                    for candidate in unvisited:

                        if (
                            self.q_table[current_state, candidate]
                            > best_q
                        ):
                            best_q = self.q_table[
                                current_state,
                                candidate
                            ]

                            best_next = candidate

                    next_state = best_next

                reward = calculate_reward(
                    self.tasks[current_state],
                    self.tasks[next_state]
                )

                future_unvisited = [
                    u for u in unvisited
                    if u != next_state
                ]

                max_future_q = max(
                    [
                        self.q_table[next_state, u]
                        for u in future_unvisited
                    ],
                    default=0
                )

                self.q_table[current_state, next_state] = (
                    (1 - RL_ALPHA)
                    * self.q_table[current_state, next_state]
                    +
                    RL_ALPHA
                    * (
                        reward
                        +
                        RL_GAMMA * max_future_q
                    )
                )

                current_state = next_state
                unvisited.remove(current_state)

    def get_best_route(self):

        best_route_indices = [0]

        unvisited = list(range(1, self.num_tasks))

        current_node = 0

        while unvisited:

            next_node = max(
                unvisited,
                key=lambda x: self.q_table[current_node, x]
            )

            best_route_indices.append(next_node)

            unvisited.remove(next_node)

            current_node = next_node

        return [self.tasks[i] for i in best_route_indices]