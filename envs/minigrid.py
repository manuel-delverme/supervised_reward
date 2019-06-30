import gym
import gym.spaces
# noinspection PyUnresolvedReferences
import gym_minigrid
import numpy as np

import config


class _MiniGrid(gym.Env):
    def __init__(self, *task_parameters):
        super().__init__()
        del task_parameters
        self._env_name = config.env_name
        self.env = gym.make(self._env_name)
        if config.max_env_steps is not None:
            self.env.max_steps = config.max_env_steps
        self.env.see_trough_walls = True
        self._observation_space = None

    def encode_observation(self, obs):
        image = obs['image'][:, :, 0]

        doors = image == 4  # door
        opens = np.logical_not(obs['image'][:, :, 2])  # door

        info_map = np.zeros(shape=(*obs['image'].shape[:-1], 4))

        # empty # floor; where you can go
        info_map[:, :, 0] = np.logical_or(image == 3, image == 1, image == 8)

        # wall and unseen are walls; where you can not go
        info_map[:, :, 1] = np.logical_or(image == 2, image == 0)

        # info_map[:, :, 0] = np.logical_or(info_map[:, :, 0], image[:, :, 0] == 3)
        # you can go in open doors
        info_map[:, :, 0][np.logical_and(doors, opens)] = 1
        # you can not go in not open doors
        info_map[:, :, 1][np.logical_and(doors, np.logical_not(opens))] = 1
        # you can interact with closed doors
        info_map[:, :, 2] = np.logical_and(doors, np.logical_not(opens))
        # goal positions
        info_map[:, :, 3] = image == 8
        assert info_map.max() == 1 and info_map.min() == 0

        # in theory 1 and 2 are complementary, so they could be skipped
        info_map[:, :, 1] = np.logical_or(image == 2, image == 0)

        assert info_map[:, :, 3].sum() <= 1

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
            self._observation_space = gym.spaces.discrete.Discrete(*ob.ravel().shape)
        return self._observation_space

    def step(self, action):
        assert 0 <= action <= 4
        if action == 3:
            action = 5
        obs, reward, done, info = self.env.step(action)
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


class MiniGrid(_MiniGrid):
    def __init__(self, *task_parameters):
        self.ob_rms = False
        super().__init__(*task_parameters)
        clipob = 10.
        cliprew = 10.
        gamma = 0.99
        epsilon = 1e-8

        self.training = True
        obs = self.reset()
        # self.ob_rms = RunningMeanStd(shape=obs.shape)
        # self.ret_rms = RunningMeanStd(shape=())
        self.ret_rms = False
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = 0
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rew, done, info = super().step(action)

        self.ret = self.ret * self.gamma + rew
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret]))
            rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)

        if done:
            self.ret = 0.
        return obs, rew, done, info

    def _obfilt(self, obs):
        if not self.ob_rms:
            return obs

        if self.training:
            self.ob_rms.update(obs)
        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        return obs

    def reset(self):
        obs = super().reset()
        return self._obfilt(obs)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def reset(self):
        """
        Reset all environments
        """
        obs = super().reset()
        filtered_obs = self._obfilt(obs)
        return filtered_obs
