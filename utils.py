import math


def euclidean_distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def calculate_reward(current_task, next_task):
    dist = euclidean_distance(
        current_task.x,
        current_task.y,
        next_task.x,
        next_task.y
    )

    if dist == 0:
        dist = 0.1

    distance_score = 100.0 / dist

    priority_score = 50.0 if next_task.priority == 1 else 10.0

    return distance_score + priority_score