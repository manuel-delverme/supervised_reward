import numpy as np
import gym.spaces
import random
from envs import discrete
import envs.gui
import enum


class GridWorldActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    # TERMINAL = -1
    # DRINK_WATER = 4
    # EAT_FOOD = 5
    # OPEN_BOX = 4


class GridWorld(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}
    _walls = {
        8: [9, ],
        14: [15, 20],

        20: [21, ],
        32: [33, ],

        12 + 0: [12 + 6, ],
        12 + 1: [12 + 7, ],
        # 12 + 2: [12 + 8, ],
        12 + 3: [12 + 9, ],
        12 + 4: [12 + 10, ],
    }
    for k, v in list(_walls.items()):
        for vi in v:
            if vi not in _walls:
                _walls[vi] = []
            _walls[vi].append(k)

    def __init__(self, side_size, terminal_states, start_from_borders=False, base_transition_probability=1.0):
        self.height = side_size
        self.width = side_size
        self.num_tiles = side_size * side_size

        transition = {}
        # self.grid = np.arange(self.height * self.width).reshape((self.height, self.width))
        # it = np.nditer(self.grid, flags=['multi_index'])

        try:
            self.terminal_positions = (terminal_states(self.num_tiles),)
        except TypeError as e:
            self.terminal_positions = terminal_states

        # Initial state distribution is uniform
        initial_state_distribution = np.ones(self.num_tiles)
        if start_from_borders:
            initial_state_distribution[[
                0, 1, 2, 3, 4, 5,
                6, 11,
                12, 12 + 5,
                18, 18 + 5,
                24, 24 + 5,
                30, 31, 32, 33, 34, 35
            ]] = 2
            initial_state_distribution -= 1

        initial_state_distribution /= initial_state_distribution.sum()

        position_change = {
            GridWorldActions.UP: - self.width,
            GridWorldActions.DOWN: + self.width,
            GridWorldActions.RIGHT: + 1,
            GridWorldActions.LEFT: - 1,
        }

        # transition probabilities
        for tile_idx in range(self.num_tiles):
            transition[tile_idx] = {}
            terminal = tile_idx in self.terminal_positions
            reward = 0.0 if terminal else -1.0

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

                if tile_idx in self._walls and s1_idx in self._walls[tile_idx]:
                    transition_probability = 0.0
                elif s1_idx in self.terminal_positions:
                    transition_probability = 0.0
                elif out_of_grid:
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

    def _render(self, mode='human', close=False):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(width=self.width)
        self.gui.print_board(
            player_position=self.agent_position_idx,
            terminal_states=self.terminal_positions,
            walls=self._walls, boxes=self.boxes
        )

    def print_board(self, some_matrix=None, close=False, policy=None):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.width)
        self.gui.print_board(
            player_position=self.agent_position_idx,
            terminal_states=self.terminal_positions,
            walls=self._walls,
            some_matrix=some_matrix,
            policy=policy,
        )

    def teleport_agent(self, new_position):
        self.agent_position_idx = new_position
