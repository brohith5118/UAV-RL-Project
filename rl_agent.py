import random


class QLearningAgent:
    def __init__(self):
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)

    def choose_action(self, state, actions):
        if random.random() < 0.2:
            return random.choice(actions)

        q_values = [self.get_q_value(state, a) for a in actions]

        max_q = max(q_values)

        for action, q in zip(actions, q_values):
            if q == max_q:
                return action

    def update(self, state, action, reward, next_state):
        old_q = self.get_q_value(state, action)

        future_q = max([
            self.get_q_value(next_state, a)
            for a in range(10)
        ], default=0)

        new_q = old_q + 0.1 * (
            reward + 0.9 * future_q - old_q
        )

        self.q_table[(state, action)] = new_q