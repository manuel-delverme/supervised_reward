import enum
import warnings
import random
import gym.spaces
import gym_minigrid
import gym_minigrid.minigrid
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
        image = obs['image'][:, :, 0]  # ignore the color and the state (for now)
        assert set(obs).issubset({1, 2, 8})
        image.flags.writeable = False
        obs = image.tostring()
        warnings.warn("TODO: make sure that turning makes the image turn")
        return obs

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def agent_position_idx(self):
        return self.env.agent_pos

    @property
    def observation_space(self):
        raise NotImplementedError

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.encode_observation(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self.encode_observation(obs)

    def render(self, **kwargs):
        renderer = self.env.render()
        return renderer

    def close(self):
        self.env.close()

    def show_board(self, *args, **kwargs):
        self.render()
