class UAV:
    def __init__(self, uav_id, x, y, battery):
        self.id = uav_id
        self.x = x
        self.y = y
        self.battery = battery
        self.tasks = []

    def position(self):
        return (self.x, self.y)

    def assign_task(self, task):
        self.tasks.append(task)

    def move_to(self, x, y, energy_cost):
        self.x = x
        self.y = y
        self.battery -= energy_cost