class Task:
    def __init__(self, task_id, x, y, priority, compute_workload):
        self.task_id = task_id
        self.x = x
        self.y = y
        self.priority = priority
        self.compute_workload = compute_workload

    def __repr__(self):
        return (
            f"Task {self.task_id}: "
            f"Loc({self.x},{self.y}) | "
            f"Pri: {self.priority} | "
            f"Load: {self.compute_workload}"
        )