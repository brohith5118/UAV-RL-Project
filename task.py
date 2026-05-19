class Task:
    def __init__(self, task_id, x, y, priority):
        self.id = task_id
        self.x = x
        self.y = y
        self.priority = priority
        self.completed = False

    def position(self):
        return (self.x, self.y)