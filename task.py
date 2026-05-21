class Task:

    def __init__(
        self,
        task_id,
        x,
        y,
        priority,
        energy_cost,
        hover_time,
        compute_load
    ):

        self.task_id = task_id
        self.x = x
        self.y = y

        self.priority = priority

        # ω(g_i)
        self.energy_cost = energy_cost
        self.hover_time = hover_time
        self.compute_load = compute_load

    def __repr__(self):

        return (
            f"Task {self.task_id} | "
            f"Loc({self.x},{self.y}) | "
            f"Priority={self.priority} | "
            f"E={self.energy_cost:.1f} | "
            f"H={self.hover_time:.1f} | "
            f"F={self.compute_load:.1f}"
        )