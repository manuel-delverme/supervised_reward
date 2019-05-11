import enum

import gym
import gym.spaces
# noinspection PyUnresolvedReferences
import gym_minigrid
import numpy as np

import config


def to_tuple(img):
    b = tuple(tuple(tuple(column) for column in row) for row in img)
    return b


class MiniGrid(gym.Env):

    def __init__(self, *task_parameters):
        del task_parameters
        self._env_name = config.env_name
        self.env = gym.make(self._env_name)
        self.env.max_steps = config.max_env_steps
        self.env.see_trough_walls = True

    def encode_observation(self, obs):
        image = obs['image']
        info_map = np.zeros(shape=(*obs['image'].shape[:-1], 4))

        # empty # floor; where you can go
        info_map[:, :, 0] = np.logical_or(image[:, :, 0] == 3, image[:, :, 0] == 1, image[:, :, 0] == 8)

        # wall and unseen are walls; where you can not go
        info_map[:, :, 1] = np.logical_or(image[:, :, 0] == 2, image[:, :, 0] == 0)

        # info_map[:, :, 0] = np.logical_or(info_map[:, :, 0], image[:, :, 0] == 3)
        doors = image[:, :, 0] == 4  # door
        opens = np.logical_not(image[:, :, 2])  # door
        # you can go in open doors
        info_map[:, :, 0][np.logical_and(doors, opens)] = 1
        # you can not go in open doors
        info_map[:, :, 1][np.logical_and(doors, np.logical_not(opens))] = 1
        # you can interact with closed doors
        info_map[:, :, 2] = np.logical_and(doors, np.logical_not(opens))
        # goal positions
        info_map[:, :, 3] = image[:, :, 0] == 8
        assert info_map.max() == 1 and info_map.min() == 0

        # in theory 1 and 2 are complementary, so they could be skipped
        return info_map

    @property
    def action_space(self):
        # pickup, drop, done are removed
        return gym.spaces.Discrete(4)

    @property
    def agent_position_idx(self):
        return np.array(self.env.agent_pos)

    @property
    def observation_space(self):
        # x, y, orientation
        return gym.spaces.discrete.Discrete(149)

    def step(self, action):
        assert 0 <= action <= 4
        if action == 3:
            action = 5
        obs, reward, done, info = self.env.step(action)
        # info['original_observation'] = obs
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
        return "ENV{" + self._env_name + "}"

    def __str__(self):
        return self.__repr__()
