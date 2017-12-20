import numpy as np

from gym import Env, spaces
from gym.utils import seeding


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(Env):
    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """

    def __init__(self, number_of_states, number_of_actions, P, initial_state_distribution):
        self.P = P
        self.initial_state_distribution = initial_state_distribution
        self.last_action = None  # for rendering
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions

        self.action_space = spaces.Discrete(self.number_of_actions)
        self.observation_space = spaces.Discrete(self.number_of_states)

        self._seed()
        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.agent_position_idx = categorical_sample(self.initial_state_distribution, self.np_random)
        self.last_action = None
        return self.agent_position_idx

    def _step(self, action):
        transitions = self.P[self.agent_position_idx][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.agent_position_idx = s
        self.last_action = action
        return s, r, d, {"prob": p}
