import numpy as np
import random
import sys
from gym.envs.toy_text import discrete
import io
import envs.gui

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
TERMINAL = -1


class GridWorld(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, size=12, terminal_state=0):
        # (UP=0, RIGHT=1, DOWN=2, LEFT=3)
        self.shape = (size, size)
        action_size = 4
        height, width = size, size
        state_size = height * width

        transition = {}
        self.grid = np.arange(state_size).reshape((height, width))
        it = np.nditer(self.grid, flags=['multi_index'])

        self.terminal_states = (terminal_state(size),)
        while not it.finished:
            state_idx = it.iterindex
            y, x = it.multi_index

            transition[state_idx] = {a: [] for a in range(action_size)}
            terminal = state_idx in self.terminal_states
            reward = 0.0 if terminal else -1.0

            if terminal:
                # We're stuck in a terminal state
                transition[state_idx][UP] = [(1.0, state_idx, reward, True)]
                transition[state_idx][RIGHT] = [(1.0, state_idx, reward, True)]
                transition[state_idx][DOWN] = [(1.0, state_idx, reward, True)]
                transition[state_idx][LEFT] = [(1.0, state_idx, reward, True)]
            else:
                # Not a terminal state
                next_state_up = state_idx if y == 0 else state_idx - width
                next_state_right = state_idx if x == (width - 1) else state_idx + 1
                next_state_down = state_idx if y == (height - 1) else state_idx + width
                next_state_left = state_idx if x == 0 else state_idx - 1
                transition[state_idx][UP] = [(1.0, next_state_up, reward, next_state_up in self.terminal_states)]
                transition[state_idx][RIGHT] = [
                    (1.0, next_state_right, reward, next_state_right in self.terminal_states)]
                transition[state_idx][DOWN] = [(1.0, next_state_down, reward, next_state_down in self.terminal_states)]
                transition[state_idx][LEFT] = [(1.0, next_state_left, reward, next_state_left in self.terminal_states)]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(state_size) / state_size

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.transition = transition
        super(GridWorld, self).__init__(state_size, action_size, transition, isd)
        self.gui = None
        self.reward_function = None

    @staticmethod
    def get_params():
        return [
            ('size', lambda: random.randint(1, 25)),
            ('terminal_state', lambda size: random.randint(1, np.prod(size))),
        ]

    def _render(self, mode='human', close=False):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.grid)
        self.gui.print_board(self.s, self.terminal_states)

    @property
    def agent_position_idx(self):
        return self.s

    def teleport_agent(self, new_state):
        self.s = new_state

    def plot_policy_and_value(self, pi, V):
        if self.gui is None:
            self.gui = envs.gui.GUI(self.grid)
        self.gui.print_board(self.s, self.terminal_states, policy=pi)

    def plot_goals(self, goals):
        if self.gui is None:
            self.gui = envs.gui.GUI(self.grid)
        self.gui.print_board(self.s, self.terminal_states, goals=goals)


if __name__ == "__main__":
    env = GridWorld()
    terminal = False
    while not terminal:
        state, reward, terminal, info = env.step(random.choice(range(4)))
        env.render(mode="ansi")
