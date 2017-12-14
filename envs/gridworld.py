import numpy as np
import sys
from gym.envs.toy_text import discrete
import io
import gui

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridWorld(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, size=12):
        # (UP=0, RIGHT=1, DOWN=2, LEFT=3)
        self.shape = (size, size)
        action_size = 4
        height, width = size, size
        state_size = height * width

        transition = {}
        grid = np.arange(state_size).reshape((height, width))
        it = np.nditer(grid, flags=['multi_index'])

        goal_state_idxs = (0, state_size - 1)
        while not it.finished:
            state_idx = it.iterindex
            y, x = it.multi_index

            transition[state_idx] = {a: [] for a in range(action_size)}
            terminal = state_idx in goal_state_idxs
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
                transition[state_idx][UP] = [(1.0, next_state_up, reward, next_state_up in goal_state_idxs)]
                transition[state_idx][RIGHT] = [(1.0, next_state_right, reward, next_state_right in goal_state_idxs)]
                transition[state_idx][DOWN] = [(1.0, next_state_down, reward, next_state_down in goal_state_idxs)]
                transition[state_idx][LEFT] = [(1.0, next_state_left, reward, next_state_left in goal_state_idxs)]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(state_size) / state_size

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.transition = transition
        super(GridWorld, self).__init__(state_size, action_size, transition, isd)
        self.gui = gui.GUI(size)

    def _render(self, mode='human', close=False):
        if close:
            return
        # self.gui.update()
        self.gui.print_board()

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        print("-" * self.shape[1] * 3)
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
        print("-" * self.shape[1] * 3)


if __name__ == "__main__":
    env = GridWorld()
    terminal = False
    while not terminal:
        state, reward, terminal, info = env.step(1)
        env.render(mode="ansi")