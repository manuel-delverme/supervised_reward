import gym
import gym.spaces
# noinspection PyUnresolvedReferences
import gym_minigrid
import numpy as np

import config
import shared.constants as C


class MiniGrid(gym.Env):
    def __init__(self):
        super().__init__()
        self._env_name = config.env_name
        self.env = gym.make(self._env_name)
        if config.max_env_steps is not None:
            self.env.max_steps = config.max_env_steps
        self.env.see_trough_walls = config.see_trough_walls
        self._observation_space = None

    def encode_observation(self, obs):
        image = obs['image'][:, :, 0]

        doors = image == 4  # door
        opens = np.logical_not(obs['image'][:, :, 2])  # door

        info_map = np.zeros(shape=(config.Minigrid.nr_layers, *obs['image'].shape[:-1]))

        # # empty # floor; where you can go
        # info_map[C.WALKABLE_LAYER, :, :] = np.logical_or(np.logical_or(image == 3, image == 1), image == 8).astype(np.int8)

        # wall and unseen are walls; where you can not go
        # info_map[C.UNWALKABLE_LAYER, :, :] = np.logical_or(image == 2, image == 0)
        info_map[C.UNWALKABLE_LAYER, :, :] = image == 2

        # # info_map[:, :, 0] = np.logical_or(info_map[:, :, 0], image[:, :, 0] == 3)
        # # you can go in open doors
        # info_map[C.WALKABLE_LAYER, :, :][np.logical_and(doors, opens)] = 1

        # you can not go in not open doors
        info_map[C.UNWALKABLE_LAYER, :, :][np.logical_and(doors, np.logical_not(opens))] = 1
        # you can interact with closed doors
        info_map[C.DOOR_LAYER, :, :] = np.logical_and(doors, np.logical_not(opens))
        # goal positions
        info_map[C.FOOD_LAYER, :, :] = image == 8

        for layer in range(info_map.shape[0]):
            info_map[layer] = np.flip(np.rot90(info_map[layer, :, :], 3), 1)

        assert info_map.max() == 1 and info_map.min() == 0
        assert info_map[C.FOOD_LAYER, :, :].sum() <= 1

        if config.blurred_observations:
            for layer in range(info_map.shape[0]):
                info_map[layer, 0:2, 0:2] = info_map[layer, 0:2, 0:2].max()
                info_map[layer, 0:2, 3:5] = info_map[layer, 0:2, 3:5].max()
                info_map[layer, 3:5, 0:2] = info_map[layer, 3:5, 0:2].max()
                info_map[layer, 3:5, 3:5] = info_map[layer, 3:5, 3:5].max()

        info_map.flags.writeable = False
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
        if self._observation_space is None:
            ob = self.reset()
            self._observation_space = gym.spaces.discrete.Discrete(*ob.reshape(-1).shape)
        return self._observation_space

    def step(self, action):
        assert 0 <= action <= 4
        if action == 3:
            action = self.env.actions.toggle
        obs, reward, done, info = self.env.step(action)

        reward = reward - 0.1
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
