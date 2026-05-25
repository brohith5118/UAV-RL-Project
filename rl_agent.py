# =========================================================
# TSA-MODULE  –  RL-based Task Sequence Adjustment
#
# Implements Algorithm 2 from the paper (Section
# "RL-based task sequence adjustment algorithm").
#
# MDP formulation (eq 27–30):
#
#   State space:
#     S = {s_0, s_1, ..., s_m, s_{m+1}}
#     s_j = (X_j, Y_j, pri_j)              (eq 27)
#     s_0 = UAV current position + resources
#     s_{m+1} = terminal (return to base)
#
#   Action space:
#     A = {a_j | a_j: s_j → s_j', j' ≠ j} (eq 28)
#     Restricted to feasibility-filtered assigned tasks.
#
#   Reward (eq 29):
#     R = c_d·d(j,j') + c_p·pri_{j'} +
#         c_t·T^re/T^max + c_c·(C^re−C_{j'})/C^max
#
#   Q-update (eq 30):
#     Q[s,a] ← (1−α)Q[s,a] + α(R + γ·max_a' Q[s',a'])
#
# Key design: compact per-UAV state-action space produced
# by the D and PR stages makes tabular Q-learning efficient
# and convergence fast (few hundred episodes).
# =========================================================

import math
import random
import numpy as np

from utils import calculate_reward, estimate_finish_time, check_deadline
from config import (
    EPOCHS,
    RL_ALPHA,
    RL_GAMMA,
    EPSILON,
    CD, CP, CT, CC,
    UAV_SPEED,
)


# ----------------------------------------------------------
# Q-LEARNING TRAJECTORY PLANNER  (per UAV)
# ----------------------------------------------------------

class QLearningTrajectoryPlanner:
    """
    Trains a Q-table over the small task-subset assigned to
    one UAV, then extracts the best execution sequence.

    Attributes
    ----------
    uav      : UAV object (position, residual resources)
    tasks    : list[Task] – the UAV's assigned task subset
    q_table  : np.ndarray shape (n_tasks, n_tasks)
               Q[i, j] = value of "go to task j when at task i"
    """

    def __init__(self, uav, tasks):

        self.uav      = uav
        self.tasks    = list(tasks)        # local copy
        self.n        = len(self.tasks)

        if self.n == 0:
            raise ValueError(
                f"UAV {uav.uav_id}: no assigned tasks for TSA."
            )

        # Index map: task_id → local index
        self.task_index = {
            t.task_id: i for i, t in enumerate(self.tasks)
        }

        # Q-table: rows = current task, cols = next task
        # Extra row/col 0 represents the UAV start position
        # Total size: (n+1) × (n+1)  where index 0 = start
        self.q_table = np.zeros(
            (self.n + 1, self.n + 1), dtype=np.float64
        )

        # Track best route found during training
        self._best_route   = None
        self._best_reward  = -float('inf')

    # --------------------------------------------------
    # REWARD  R(s_j, a_j)  (eq 29)
    # --------------------------------------------------

    def _reward(self, from_task_or_pos, to_task_idx):
        """
        Compute reward for transitioning to tasks[to_task_idx].
        from_task_or_pos is either a Task or (x, y) tuple.
        """
        next_task = self.tasks[to_task_idx]
        return calculate_reward(
            from_task_or_pos,
            next_task,
            self.uav,
            CD, CP, CT, CC,
        )

    # --------------------------------------------------
    # FEASIBILITY FILTER
    #
    # A task is feasible as "next" only if the UAV has
    # enough residual hover-time to reach it and return.
    # (Preserves the paper's feasibility-filtered action
    # space that keeps Q-learning tractable.)
    # --------------------------------------------------

    def _feasible_actions(self, visited, current_pos):
        """
        Returns indices of unvisited tasks that are still
        reachable (hover-time feasibility, eq 22).
        """
        feasible = []
        for idx in range(self.n):
            if idx in visited:
                continue
            task = self.tasks[idx]
            dist = math.hypot(
                current_pos[0] - task.x,
                current_pos[1] - task.y,
            )
            travel_t = dist / UAV_SPEED
            # rough remaining capacity check
            if travel_t + task.hover_time <= self.uav.remaining_hover_time:
                feasible.append(idx)
        # If nothing feasible, open all unvisited (graceful fallback)
        if not feasible:
            feasible = [i for i in range(self.n) if i not in visited]
        return feasible

    # --------------------------------------------------
    # EPSILON-GREEDY ACTION SELECTION
    # --------------------------------------------------

    def _select_action(self, state_idx, feasible_actions, epsilon):
        if not feasible_actions:
            return None
        if random.random() < epsilon:
            return random.choice(feasible_actions)
        # Greedy: max Q over feasible actions
        q_vals = [
            (self.q_table[state_idx, a], a)
            for a in feasible_actions
        ]
        return max(q_vals, key=lambda x: x[0])[1]

    # --------------------------------------------------
    # SINGLE EPISODE
    # --------------------------------------------------

    def _run_episode(self, epsilon):
        """
        One full Q-learning episode.
        Returns total episode reward.
        """
        # State index 0 = UAV start; task states are 1..n
        current_pos   = (self.uav.x, self.uav.y)
        current_state = 0          # start node
        visited       = set()
        episode_reward = 0.0
        route_indices  = []

        while len(visited) < self.n:

            feasible = self._feasible_actions(visited, current_pos)
            if not feasible:
                break

            # Action = next task local index (1-based in q_table)
            action = self._select_action(
                current_state, feasible, epsilon
            )
            if action is None:
                break

            next_task = self.tasks[action]
            r         = self._reward(
                current_pos if current_state == 0
                else self.tasks[current_state - 1],
                action
            )

            # Max future Q over remaining unvisited tasks
            next_visited  = visited | {action}
            next_feasible = [
                i for i in range(self.n)
                if i not in next_visited
            ]
            if next_feasible:
                max_future_q = max(
                    self.q_table[action + 1, j + 1]
                    for j in next_feasible
                )
            else:
                max_future_q = 0.0

            # Q-update  (eq 30)
            old_q = self.q_table[current_state, action + 1]
            self.q_table[current_state, action + 1] = (
                (1 - RL_ALPHA) * old_q
                + RL_ALPHA * (r + RL_GAMMA * max_future_q)
            )

            episode_reward += r
            visited.add(action)
            route_indices.append(action)
            current_pos   = (next_task.x, next_task.y)
            current_state = action + 1

        return episode_reward, route_indices

    # --------------------------------------------------
    # TRAINING  (Algorithm 2)
    # --------------------------------------------------

    def train(self, epochs=EPOCHS, verbose=False):
        """
        Run Q-learning for *epochs* episodes with decaying
        epsilon (exploration → exploitation).
        """
        epsilon     = EPSILON
        eps_decay   = epsilon / max(epochs * 0.8, 1)

        reward_log  = []

        for ep in range(epochs):

            ep_reward, route = self._run_episode(epsilon)
            reward_log.append(ep_reward)

            # Track best complete route seen
            if (len(route) == self.n
                    and ep_reward > self._best_reward):
                self._best_reward = ep_reward
                self._best_route  = route[:]

            # Decay exploration
            epsilon = max(0.01, epsilon - eps_decay)

        if verbose:
            mean_r = sum(reward_log[-50:]) / min(50, len(reward_log))
            print(
                f"  UAV {self.uav.uav_id} TSA: "
                f"{epochs} eps, "
                f"last-50 mean reward={mean_r:.2f}, "
                f"best route reward={self._best_reward:.2f}"
            )

        return reward_log

    # --------------------------------------------------
    # EXTRACT BEST ROUTE
    # --------------------------------------------------

    def get_best_route(self):
        """
        Return the task execution sequence as an ordered
        list of Task objects.

        Uses the best route recorded during training.
        Falls back to greedy Q-table extraction if no
        complete route was found.
        """

        if self._best_route and len(self._best_route) == self.n:
            return [self.tasks[i] for i in self._best_route]

        # --- Greedy fallback ---
        # Start from the task nearest to the UAV
        start = min(
            range(self.n),
            key=lambda i: math.hypot(
                self.tasks[i].x - self.uav.x,
                self.tasks[i].y - self.uav.y,
            )
        )

        route   = [start]
        visited = {start}
        current = start + 1    # q_table index (1-based)

        while len(visited) < self.n:
            remaining = [
                j for j in range(self.n) if j not in visited
            ]
            if not remaining:
                break
            next_idx = max(
                remaining,
                key=lambda j: self.q_table[current, j + 1]
            )
            route.append(next_idx)
            visited.add(next_idx)
            current = next_idx + 1

        return [self.tasks[i] for i in route]

    # --------------------------------------------------
    # DEADLINE-AWARE SEQUENCE REORDER
    #
    # Post-processes the Q-learned route to push any
    # urgent (priority-1) tasks earlier if deadline risk
    # is detected.  Mirrors the TSA "task-sequence dynamic
    # adjustment" described in paper Section 3.4.
    # --------------------------------------------------

    def reorder_by_deadline(self, route):
        """
        Given an initial route (list of Tasks), move any
        task whose deadline would be missed to the earliest
        feasible position.

        Returns the adjusted route.
        """
        route    = list(route)
        adjusted = True

        while adjusted:
            adjusted  = False
            timeline  = estimate_finish_time(
                self.uav, route, UAV_SPEED
            )

            for rank, (task, ft) in enumerate(timeline):
                if check_deadline(task, ft):
                    continue
                # Task is at risk – try moving it earlier
                best_pos = rank
                for pos in range(rank):
                    candidate = route[:pos] + [task] + \
                                route[pos:rank] + route[rank + 1:]
                    tl2 = estimate_finish_time(
                        self.uav, candidate, UAV_SPEED
                    )
                    _, ft2 = tl2[pos]
                    if check_deadline(task, ft2):
                        best_pos = pos
                        break
                if best_pos != rank:
                    route.pop(rank)
                    route.insert(best_pos, task)
                    adjusted = True
                    break   # restart scan

        return route


# ----------------------------------------------------------
# FLEET-LEVEL TSA  (run planner for every UAV)
# ----------------------------------------------------------

def run_tsa_for_fleet(uavs, epochs=EPOCHS, verbose=True):
    """
    Trains a Q-learning planner for each UAV in the fleet
    and returns a dict  {uav_id: ordered_task_list}.

    Applies deadline-aware reordering after Q-learning.
    """
    all_routes = {}

    for uav in uavs:

        if not uav.active:
            all_routes[uav.uav_id] = []
            continue

        if not uav.assigned_tasks:
            if verbose:
                print(f"  UAV {uav.uav_id:02d}: no tasks assigned.")
            all_routes[uav.uav_id] = []
            continue

        if verbose:
            print(f"  Training TSA for UAV {uav.uav_id:02d} "
                  f"({len(uav.assigned_tasks)} tasks)...")

        planner = QLearningTrajectoryPlanner(uav, uav.assigned_tasks)
        planner.train(epochs=epochs, verbose=verbose)

        route = planner.get_best_route()
        route = planner.reorder_by_deadline(route)

        all_routes[uav.uav_id] = route

    return all_routes