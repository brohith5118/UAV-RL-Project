class UAV:

    def __init__(
        self,
        uav_id,
        x,
        y,
        max_energy,
        max_hover_time,
        max_compute
    ):

        self.uav_id = uav_id

        self.x = x
        self.y = y

        # Capacity vector C_u(t)
        self.max_energy = max_energy
        self.max_hover_time = max_hover_time
        self.max_compute = max_compute

        self.remaining_energy = max_energy
        self.remaining_hover_time = max_hover_time
        self.remaining_compute = max_compute

        # Lagrange multipliers μ_u,k
        self.mu_energy = 0.0
        self.mu_hover = 0.0
        self.mu_compute = 0.0

        self.assigned_tasks = []

    def clear_tasks(self):

        self.assigned_tasks = []

    def reset_resources(self):

        self.remaining_energy = self.max_energy
        self.remaining_hover_time = self.max_hover_time
        self.remaining_compute = self.max_compute