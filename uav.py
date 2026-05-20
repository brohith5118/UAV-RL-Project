class UAV:
    def __init__(self, uav_id, x, y, max_energy, max_compute):
        self.uav_id = uav_id
        self.x = x
        self.y = y
        self.max_energy = max_energy
        self.max_compute = max_compute

        self.penalty_energy = 0.0
        self.penalty_compute = 0.0

        self.assigned_tasks = []

    def clear_tasks(self):
        self.assigned_tasks = []