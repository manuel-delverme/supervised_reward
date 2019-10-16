import collections
from typing import Text, List, Union, Tuple

import numpy as np

import config


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
    def __init__(self, target_state: List[Union[Tuple, Text]], reward_partials=True):
        assert isinstance(target_state, list)
        self.target_state = target_state

        self.fitness = None
        self.options = None
        self.is_option = not reward_partials
        self.reset()

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
                if s[2]:
                    str_k = f"{s[0]} {s[1]}"
                else:
                    str_k = f"{s[0]} ¬{s[1]}"
            step.append(str_k)
        return f"reach {step}"

    def __call__(self, new_state, environment, as_inhibition=False):
        self._update_state(environment)

        print(f'reward: is_option {self.is_option}', self.achieved_ltl_clauses, self.target_state)
        # reward = config.option_termination_treshold
        # TODO: options used to reward {0, 1} now they need to do proper initibition so that the convergence criteria will bug out

        reward = len(self.achieved_ltl_clauses)  # +1 for each node
        if reward:
            if self.is_option and not as_inhibition:
                reward = config.option_trigger_treshold  # TODO: this could cause issues with the initibiotn
        else:
            reward = -0.01

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
        for preprosition, value in state.items():
            self.state_history[preprosition].append(value)

        new_ltl_clauses = set()
        # don't get to the door until you get the key,
        # ~FD until+K AND (OD and last+FD)

        achieved_subproblems = []
        evaluation1 = []
        for idx, step in enumerate(self.target_state):
            if isinstance(step, (tuple, list)):
                operator, preprosition, target_value = step
                is_satisfied = False
                is_true = self.last_state[preprosition] == target_value
                was_always_true = all(v == target_value for v in self.state_history[preprosition])
                has_been_true = any(v == target_value for v in self.state_history[preprosition])

                if operator == "last":
                    is_satisfied = is_true
                elif operator == "global":
                    is_satisfied = was_always_true
                elif operator == "past":
                    is_satisfied = has_been_true

                evaluation1.append(is_satisfied)
                if is_satisfied:
                    achieved_subproblems.append(idx)
            else:
                evaluation1.append(step)

        evaluation2 = []
        for idx in range(1, len(evaluation1), 2):
            step = evaluation1[idx - 1: idx + 2]

            a, operator, b = step
            is_satisfied = False

            if operator == "until":  # not door until key
                if not b:  # not key
                    is_satisfied = a  # not door
                else:
                    is_satisfied = True  # not active TODO: continue here

            elif operator == "and":
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
        self.state_history = collections.defaultdict(list)
        self.achieved_ltl_clauses = []

    def motivating_function(self, _):
        # motivating_trace = []
        # for idx, target_state in enumerate(self.target_state):
        #     if idx not in self.achieved_ltl_clauses:
        #         if isistance(target_state, ) == 3:
        #             operator, preprosition, target_value = target_state
        #             target_state = operator, preprosition, None
        #     motivating_trace.append(target_state)
        motivating_trace = []
        if len(self.achieved_ltl_clauses) > 3:
            raise NotImplementedError

        for idx in self.achieved_ltl_clauses:
            symbol = self.target_state[idx]
            motivating_trace.append(symbol)

            if isinstance(symbol, str):
                if symbol == 'until':
                    if (idx + 1) not in self.achieved_ltl_clauses:
                        # it's important as a form but not as an independent symbol
                        motivating_trace.append(self.target_state[idx + 1])

        return LTLReward(motivating_trace, reward_partials=False)
