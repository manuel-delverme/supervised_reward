import collections
from typing import Text, List, Union, Tuple

import numpy as np

import config
from learners.helpers import CachedPolicy


class Reward:
    def __init__(self, reward_vector):
        self.reward_vector = reward_vector
        self.fitness = None
        self.options = None

    def __call__(self, observation, environment):
        image = observation.reshape(-1)
        return self.reward_vector.dot(image)

    def __str__(self):
        # return str(self.reward_vector.reshape(-1, config.agent_view_size, config.agent_view_size))
        return str(np.where(self.reward_vector > 0))

    def __repr__(self):
        return str(self)

    def reset(self):
        pass

    def motivating_function(self, state):
        motivating_function = np.multiply(self.reward_vector, state.reshape(-1))
        negative_rewards = self.reward_vector[self.reward_vector < 0]
        # keep the punishments
        motivating_function[self.reward_vector < 0] = negative_rewards

        return MotivatingFunction(motivating_function)


class MotivatingFunction(Reward):
    def __repr__(self):
        representation = self.reward_vector.reshape(-1, config.agent_view_size, config.agent_view_size)
        # for layer in range(representation.shape[0]):
        #     representation[layer]
        # return repr()
        return repr(representation)


class ConstrainedReward(Reward):
    def __init__(self, reward_vector, normalize=True):
        self.reward_parameters = reward_vector
        self.reward_coords = []
        points = reward_vector.reshape(-1, 3)
        reward = np.ones(shape=(config.Minigrid.nr_layers, config.agent_view_size, config.agent_view_size)) * -0.005
        for layer, x, y in points:
            if normalize:
                x *= config.agent_view_size / 2
                x += config.agent_view_size / 2
                x = np.clip(x, 0, config.agent_view_size - 1)

                y *= config.agent_view_size / 2
                y += config.agent_view_size / 2
                y = np.clip(y, 0, config.agent_view_size - 1)

                layer *= config.Minigrid.nr_layers
                layer += config.Minigrid.nr_layers / 2
                layer = np.clip(layer, 0, config.Minigrid.nr_layers - 1)
            reward[int(layer), int(x), int(y)] = 2
            self.reward_coords.extend([int(layer), int(x), int(y)])
        super().__init__(reward.reshape(-1))

    def mutate(self):
        # return self.reward_parameters + np.random.randn(*self.reward_parameters.shape) / 10
        new_params = np.array(self.reward_coords)
        new_params[np.random.randint(0, len(self.reward_coords))] += np.random.choice((-1, 1))
        return new_params


class LTLReward:
    available_actions: List[CachedPolicy]

    def __init__(self, target_state: List[Union[Tuple, Text]], available_actions: List[CachedPolicy] = (), ):
        assert isinstance(target_state, list)
        self.target_state = target_state

        self.fitness = None
        self.options = None
        self.available_actions = [a for a in available_actions if hasattr(a, 'motivating_function')]
        self.reset()

    def add_new_action(self, action):
        assert hasattr(action, 'motivating_function')
        self.available_actions.append(action)

    @property
    def reward_coords(self):
        return repr(self)

    def complete(self):
        return len(self.target_state) == self.ltl_progress

    def __repr__(self):
        step = []
        for s in self.target_state:
            str_k = s
            if isinstance(s, (tuple, list)):
                s0 = s[0].upper()
                if s[1]:
                    str_k = f"{s0}"
                else:
                    str_k = f"¬{s0}"
            step.append(str_k)
        step = " ".join(step)
        return f"reach [{step}]"

    def __call__(self, new_state, environment):
        self._update_state(environment)

        # if not self.is_option:
        #     print(f'reward: is_option {self.is_option}')
        #     print(f'ACHIEVED: {[self.target_state[idx] for idx in self.achieved_ltl_clauses]}')
        #     print(f'OF: {self.target_state}')
        # if type_of_run in ('discovery', 'visualization'):
        # for option in inibited_rewards:
        #     inibition = option.motivating_function(new_state, environment)
        #     reward -= inibition

        # reward = config.option_termination_treshold
        # TODO: options used to reward {0, 1} now they need to do proper initibition so that the convergence criteria will bug out

        reward = len(self.achieved_ltl_clauses)  # +1 for each node
        inhibitions = 0.0
        for action in self.available_actions:
            inhibition = action.motivating_function(new_state, environment)
            inhibitions += inhibition
        reward -= inhibitions

        return float(reward)

    def _update_state(self, environment):
        grid = environment.env.grid
        state = {
            'key': True  # key is in hand unless it's on the ground
        }

        for idx, cell in enumerate(grid.grid):
            if cell is None:
                continue
            if cell.type == 'wall':
                continue

            if cell.type == 'key':
                state['key'] = False
            elif cell.type == 'door':
                state['door'] = cell.is_open

            elif cell.type == 'goal':
                y, x = divmod(idx, environment.env.width)
                state['goal'] = all(environment.env.agent_pos == (y, x))
            else:
                raise Exception(cell)
        front_cell = environment.env.grid.get(*environment.env.front_pos)
        state['front_door'] = front_cell is not None and front_cell.type == "door"

        self.last_state = state

        new_ltl_clauses = set()
        # don't get to the door until you get the key,
        # ~FD until+K AND (OD and last+FD)

        achieved_subproblems = []
        evaluation1 = []
        for idx, step in enumerate(self.target_state):
            if isinstance(step, (tuple, list)):
                preprosition, target_value = step
                is_true = self.last_state[preprosition] == target_value

                evaluation1.append(is_true)
                if is_true:
                    achieved_subproblems.append(idx)
            else:
                evaluation1.append(step)

        evaluation2 = []
        for idx in range(1, len(evaluation1), 2):
            step = evaluation1[idx - 1: idx + 2]

            a, operator, b = step
            is_satisfied = False

            if operator == "and":
                is_satisfied = a and b
            elif operator == "or":
                is_satisfied = a or b

            if is_satisfied:
                achieved_subproblems.append(idx)
            evaluation2.append(is_satisfied)

        self.achieved_ltl_clauses = achieved_subproblems

    def reset(self):
        self.ltl_progress = 0
        self.last_state = {}
        self.achieved_ltl_clauses = []

    def motivating_function(self, _state, available_actions):
        print(f"Generating options for {[self.target_state[idx] for idx in sorted(self.achieved_ltl_clauses)]}")

        motivating_trace = []
        for idx in sorted(self.achieved_ltl_clauses):
            symbol = self.target_state[idx]
            motivating_trace.append(symbol)

        return LTLReward(motivating_trace, available_actions)
