import envs.discrete
import enum
import numpy as np


class GridWorldActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class GridWorld(envs.discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    _walls = {10, 17, 21, 22, 23, 24, 25, 26, 31, 45}

    def __init__(self, side_size, terminal_states, base_transition_probability=0.9):
        self.height = side_size
        self.width = side_size
        self.num_tiles = side_size * side_size

        transition = {}
        self.terminal_positions = terminal_states

        # Initial state distribution is uniform
        initial_state_distribution = np.ones(self.num_tiles)
        initial_state_distribution[list(self._walls)] = 0
        initial_state_distribution /= initial_state_distribution.sum()

        position_change = {
            GridWorldActions.UP: - self.width,
            GridWorldActions.DOWN: + self.width,
            GridWorldActions.RIGHT: + 1,
            GridWorldActions.LEFT: - 1,
        }

        # transition probabilities
        reward = -1.0
        for tile_idx in range(self.num_tiles):
            transition[tile_idx] = {}

            x = tile_idx % side_size
            y = tile_idx // side_size

            for action in GridWorldActions:
                out_of_grid = (
                    (action == GridWorldActions.UP and y == 0) or
                    (action == GridWorldActions.LEFT and x == 0) or
                    (action == GridWorldActions.RIGHT and x == (self.width - 1)) or
                    (action == GridWorldActions.DOWN and y == (self.height - 1))
                )
                s1_idx = tile_idx + position_change[action]

                if out_of_grid or s1_idx in self._walls:
                    transition_probability = 0.0
                else:
                    transition_probability = base_transition_probability

                transition[tile_idx][action.value] = [
                    (transition_probability, s1_idx, reward, s1_idx in self.terminal_positions),
                    (1.0 - transition_probability, tile_idx, reward, tile_idx in self.terminal_positions),
                ]

        # init discrete env
        super(GridWorld, self).__init__(
            side_size * side_size, len(list(GridWorldActions)),
            transition, initial_state_distribution
        )
        self.gui = None
        self.reward_function = None

