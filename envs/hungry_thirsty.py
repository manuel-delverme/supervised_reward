import enum
import random

import gym.spaces
import numpy as np

import envs.gridworld


class HungryThirstyActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    DRINK_WATER = 4
    EAT_FOOD = 5


class HungryThirsty(envs.gridworld.GridWorld):
    def __init__(self, box1, box2, side_size):
        self._state = {
            'hungry': True,
            'thirsty': True,
        }
        super(HungryThirsty, self).__init__(
            side_size=side_size,
            terminal_states=(),
            base_transition_probability=0.9,
        )
        extra_dof = len(self._state) * 2
        self.water_position, self.food_position = box1, box2
        self.observation_space = gym.spaces.Discrete(self.observation_space.n * extra_dof)
        self.action_space = gym.spaces.Discrete(self.action_space.n + len(self._state))

    def _step(self, action):
        action = HungryThirstyActions(action)
        terminal = False
        info = {}
        self._state['hungry'] = True

        if action == HungryThirstyActions.EAT_FOOD:
            if self.agent_position_idx == self.food_position and not self._state['thirsty']:
                self._state['hungry'] = False
        elif action == HungryThirstyActions.DRINK_WATER:
            if self.agent_position_idx == self.water_position:
                self._state['thirsty'] = False
        else:
            tile_idx, reward, terminal, info = super(envs.gridworld.GridWorld, self)._step(action.value)

        state_obj = self._hash_state()

        reward = -1
        if not self._state['hungry']:
            reward = 100
            self._state['thirsty'] = True
        return state_obj, reward, terminal, info

    def _reset(self):
        _ = super(envs.gridworld.GridWorld, self)._reset()
        self._state['hungry'] = True
        self._state['thirsty'] = random.choice((True, False))
        return self._hash_state()

    def _hash_state(self):
        state_hash = self.agent_position_idx
        state_hash += self._state['hungry'] * self.number_of_tiles
        state_hash += self._state['thirsty'] * self.number_of_tiles * 2
        return state_hash  # state.State(state_hash=state_hash, state_info=self._state.copy())

    def show_board(self, some_matrix=None, close=False, policy=None, highlight_square=None, info={}, option_vec=(),
                   highlight_squares=(), state_offset=None):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.width)
        info.update({
            'hungry': self._state['hungry'],
            'thirsty': self._state['thirsty'],
        })
        action_names = [o.index(-1) for o in option_vec]
        if state_offset is None:
            state_offset = self.number_of_tiles * (self._hash_state() // self.number_of_tiles)

        self.gui.render_board(
            player_position=self.agent_position_idx,
            terminal_states=self.terminal_positions,
            water_position=self.water_position,
            food_position=self.food_position,
            walls=self._walls,
            some_matrix=some_matrix,
            policy=policy,
            highlight_square=highlight_square,
            highlight_squares=highlight_squares,
            info=info,
            state_offset=state_offset,
            action_names=action_names,
        )

    def force_state(self, state):
        position, state_dict = self._decode_state(state)
        self._state = state_dict
        super(HungryThirsty, self).teleport_agent(position)

    def _decode_state(self, state_hash):
        # state = pos + hungry * tiles + thirsty * tiles * 2
        state_dict = {}
        if state_hash >= self.number_of_tiles * 2:
            state_hash -= self.number_of_tiles * 2
            state_dict['thirsty'] = True
        else:
            state_dict['thirsty'] = False

        if state_hash >= self.number_of_tiles:
            state_dict['hungry'] = True
            state_hash -= self.number_of_tiles
        elif state_hash < self.number_of_tiles:
            state_dict['hungry'] = False
        return state_hash, state_dict

    def render(self, mode='human'):
        if mode == 'ascii':
            board = np.zeros((self.height, self.width))
            y, x = divmod(self.agent_position_idx, self.height)
            board[y, x] = 1
            for w in self._walls:
                board[divmod(w, self.height)] = 255
            return board
        else:
            super(HungryThirsty, self).render(mode)
