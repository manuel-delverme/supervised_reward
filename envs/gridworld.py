import numpy as np
import random
from envs import discrete
import envs.gui
import enum


class Actions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    # TERMINAL = -1
    OPEN_BOX = 4
    DRINK_WATER = 5
    EAT_FOOD = 6


class GridWorld(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}
    _walls = [
        # vertical
        (7, 8,),
        (14, 15,),
        (19, 20,),
        (32, 33,),

        # horizontal
        (12 + 0, 12 + 6,),
        (12 + 1, 12 + 6,),
        (12 + 2, 12 + 7,),
        (12 + 3, 12 + 8,),
        (12 + 4, 12 + 9,),
        (12 + 5, 12 + 10,),
    ]

    def __init__(self, size, terminal_states, box_positions=(0, 5, 6 * 5 - 1, 6 * 6 - 1), num_boxes=2,
                 domain='Hungry-Thirsty'):
        self.domain = domain
        self._state = {
            'hungry': True,
            'thirsty': True,
        }

        self.shape = (size, size)
        height, width = size, size
        board_size = height * width

        if domain == 'Hungry-Thirsty':
            action_size = 6
            state_size = board_size * 2 * len(self._state)
        else:
            raise NotImplementedError()

        transition = {}
        self.grid = np.arange(board_size).reshape((height, width))
        it = np.nditer(self.grid, flags=['multi_index'])

        try:
            self.terminal_positions = (terminal_states(size),)
        except TypeError as e:
            self.terminal_positions = terminal_states

        base_transition_probability = 0.9

        # Initial state distribution is uniform
        initial_state_distribution = np.ones(board_size)

        # self.transition = transition
        self.boxes = []
        for box_idx in random.sample(box_positions, num_boxes):
            initial_state_distribution[box_idx] = 0
            self.boxes.append(box_idx)

        initial_state_distribution /= np.sum(board_size)

        position_change = {
            Actions.UP: - width,
            Actions.DOWN: + width,
            Actions.RIGHT: + 1,
            Actions.LEFT: - 1,

            Actions.EAT_FOOD: 0,
            Actions.OPEN_BOX: 0,
            Actions.DRINK_WATER: 0,
        }
        # transition probabilities
        while not it.finished:
            position_idx = it.iterindex
            y, x = it.multi_index

            transition[position_idx] = {a: [] for a in Actions}
            terminal = position_idx in self.terminal_positions
            reward = 0.0 if terminal else -1.0

            for action in Actions:
                out_of_grid = (
                    (action == Actions.UP and y == 0) or
                    (action == Actions.LEFT and x == 0) or
                    (action == Actions.RIGHT and x == (width - 1)) or
                    (action == Actions.DOWN and y == (height - 1))
                )
                s1_idx = position_idx + position_change[action]

                if tuple(sorted((position_idx, s1_idx))) in self._walls:
                    transition_probability = 0
                elif s1_idx in self.boxes:
                    transition_probability = 0
                elif s1_idx in self.terminal_positions:
                    transition_probability = 0
                elif out_of_grid:
                    transition_probability = 0
                else:
                    transition_probability = base_transition_probability

                transition[position_idx][action] = [
                    (transition_probability, s1_idx, reward, s1_idx in self.terminal_positions),
                    (1.0 - transition_probability, position_idx, reward, position_idx in self.terminal_positions),
                ]
            # overwrite actions
            transition[position_idx][Actions.OPEN_BOX] = [(1.0, position_idx, reward, False)]
            transition[position_idx][Actions.EAT_FOOD] = [(1.0, position_idx, reward, False)]
            transition[position_idx][Actions.DRINK_WATER] = [(1.0, position_idx, reward, False)]
            it.iternext()

        super(GridWorld, self).__init__(state_size, action_size, transition, initial_state_distribution)
        self.gui = None
        self.reward_function = None

    def _step(self, action):
        action = Actions(action)
        state_idx, reward, terminal, info = super(GridWorld, self)._step(action)
        self._state['hungry'] = True
        if random.random() < 0.1:
            self._state['thirsty'] = True
        if action == Actions.EAT_FOOD:
            assert self.domain in ('Hungry-Thirsty',)
            if self._state['thristy']:
                self._state['hungry'] = False
        elif action == Actions.DRINK_WATER:
            assert self.domain in ('Hungry-Thirsty',)
            self._state['thirsty'] = False
        elif action == Actions.OPEN_BOX:
            pass
        return state_idx, reward, terminal, info

    @staticmethod
    def get_params():
        return [
            ('size', lambda: random.randint(1, 25)),
            # ('terminal_states', lambda size: random.randint(1, np.prod(size))),
            ('terminal_states', tuple()),
        ]

    def _render(self, mode='human', close=False):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.grid)
        self.gui.print_board(self.agent_position_idx, self.terminal_positions)

    def teleport_agent(self, new_state):
        self.agent_position_idx = new_state

    def plot_policy_and_value(self, pi, V):
        if self.gui is None:
            self.gui = envs.gui.GUI(self.grid)
        self.gui.print_board(self.agent_position_idx, self.terminal_positions, policy=pi)

    def plot_goals(self, goals):
        if self.gui is None:
            self.gui = envs.gui.GUI(self.grid)
        self.gui.print_board(self.agent_position_idx, self.terminal_positions, goals=goals)
