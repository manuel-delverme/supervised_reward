import gym_minigrid
import gym

import collections
import enum
import warnings

import gym.spaces
import numpy as np


def to_tuple(img):
    b = tuple(tuple(tuple(column) for column in row) for row in img)
    return b


class BoxWorldActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


image_offsets = 2 ** np.arange(5 * 5).reshape(5, 5)


class MiniGrid(gym.Env):
    def __init__(self, *task_parameters):
        del task_parameters
        self.env = gym.make('MiniGrid-Empty-6x6-v0')

    def encode_observation(self, obs):
        return tuple([*self.env.agent_pos, obs['direction']])

    @property
    def action_space(self):
        # return self.env.action_space
        # pickup, drop, toggle, done are removed
        return gym.spaces.Discrete(3)

    @property
    def agent_position_idx(self):
        return self.env.agent_pos

    @property
    def observation_space(self):
        # x, y, orientation
        return gym.spaces.discrete.Discrete(3)

    def step(self, action):
        assert 0 <= action <= 3
        obs, reward, done, info = self.env.step(action)
        info['original_observation'] = obs
        # reward = reward-sum((abs(a) for a in (self.env.agent_pos[0] - 4, self.env.agent_pos[1] - 4))) / 10
        return self.encode_observation(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.encode_observation(obs)

    def render(self, *args, **kwargs):
        renderer = self.env.render(*args, **kwargs)
        return renderer

    def close(self):
        self.env.close()

    def show_board(self, *args, **kwargs):
        self.render()

    def __repr__(self):
        return 'MiniGrid-Empty-6x6-v0'
