import enum
import envs.gridworld
import envs.gui
import random
import gym.spaces


class BoxWorldActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class BoxWorld(envs.gridworld.GridWorld):
    def __init__(self, box1, box2, side_size):

        super(BoxWorld, self).__init__(
            side_size=side_size,
            terminal_states=(),
            base_transition_probability=0.9,
        )
        extra_dof = 2 * 2  # 2 boxes in 2 states.
        self.box1, self.box2 = box1, box2
        self.box1_full, self.box2_full = True, True

        self.observation_space = gym.spaces.Discrete(self.observation_space.n * extra_dof)
        self.action_space = gym.spaces.Discrete(len(BoxWorldActions))

    def _step(self, _action):
        action = BoxWorldActions(_action)

        terminal = False
        reward = -1.0

        if self.agent_position_idx == self.box1 and self.box1_full:
            self.box1_full = False
            reward = 100
            terminal = True

        elif self.agent_position_idx == self.box2 and self.box2_full:
            self.box2_full = False
            reward = 100
            terminal = True

        # open boxes randomly close
        self.box1_full = True if random.random() < 0.1 else self.box1_full
        self.box2_full = True if random.random() < 0.1 else self.box2_full

        tile_idx, _, _, info = super(envs.gridworld.GridWorld, self)._step(action.value)
        state_obj = self._hash_state()

        return state_obj, reward, terminal, info

    def _hash_state(self):
        state_hash = self.agent_position_idx
        state_hash += self.box1_full * self.number_of_tiles
        state_hash += self.box2_full * self.number_of_tiles * 2
        return state_hash

    def reset(self):
        super(BoxWorld, self).reset()
        self.box1_full = self.box2_full = True
        return self._hash_state()

    def show_board(self, some_matrix=None, close=False, policy=None, highlight_square=None, just_numbers=True):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.width)
        info = self._state.copy()
        self.gui.render_board(
            player_position=self.agent_position_idx,
            terminal_states=self.terminal_positions,
            boxes=self._state['box'],
            walls=self._walls,
            thirsty=False,  # there is no thirsty in boxes
            hungry=True,
            some_matrix=some_matrix,
            policy=policy,
            highlight_square=highlight_square,
            info=info,
            state_offset=self.number_of_tiles * (self._hash_state() // self.number_of_tiles),
            # just_numbers=True
        )

    def force_state(self, state):
        # TODO: remove, will cause bugs
        super(BoxWorld, self).teleport_agent(state % self.num_tiles)
