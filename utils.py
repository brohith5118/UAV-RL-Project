import math


def euclidean_distance(x1, y1, x2, y2):

    return math.hypot(x1 - x2, y1 - y2)


# RL trajectory reward
# Higher priority + lower movement cost

def calculate_reward(current_position, next_task):

    dist = euclidean_distance(
        current_position[0],
        current_position[1],
        next_task.x,
        next_task.y
    )

    priority_reward = (
        120 if next_task.priority == 1 else 40
    )

    movement_penalty = dist * 2

    hover_penalty = next_task.hover_time * 0.5

    compute_penalty = next_task.compute_load * 0.1

    return (
        priority_reward
        - movement_penalty
        - hover_penalty
        - compute_penalty
    )