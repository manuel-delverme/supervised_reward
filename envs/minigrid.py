import enum
import random
import gym.spaces
import gym_minigrid
import gym_minigrid.minigrid


def to_tuple(img):
    b = tuple(tuple(tuple(column) for column in row) for row in img)
    return b


class BoxWorldActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class MiniGrid(gym.Env):
    def __init__(self, *task_parameters):
        del task_parameters
        self.env = gym.make('MiniGrid-Empty-6x6-v0')
        # env = FullyObsWrapper(env)
        # self.env.max_steps = min(self.env.max_steps, 200)
        self.id_to_observation = []
        self.id_to_observation_set = set()
        self.observation_to_id_dict = dict()

    def encode_observation(self, obs):
        a = obs['image']
        a.flags.writeable = False

        obs = (a.tostring(), obs['direction'])

        if obs not in self.id_to_observation_set:
            self._add_encoding(obs)
        idx = self.observation_to_id_dict[obs]
        return idx

    def _add_encoding(self, obs):
        self.id_to_observation_set.add(obs)
        self.id_to_observation.append(obs)
        self.observation_to_id_dict[obs] = len(self.id_to_observation) - 1

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def agent_position_idx(self):
        return self.env.agent_pos

    @property
    def observation_space(self):
        possible_states = 19600  # not really sure about this
        possible_states *= 10  # safety
        return gym.spaces.Discrete(possible_states)

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

