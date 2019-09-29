from typing import Text, List, Dict

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
    def __init__(self, target_state: List[Dict[Text, bool]], reward_partials=True):
        assert isinstance(target_state, list)
        assert isinstance(target_state[0], dict)
        assert all(t for t in target_state)

        if len(target_state) > 1:
            raise NotImplementedError

        self.target_state = target_state

        self.fitness = None
        self.options = None
        self.reward_partials = reward_partials
        self.reset()

    @property
    def reward_coords(self):
        return repr(self)

    def complete(self):
        return len(self.target_state) == self.ltl_progress

    def __repr__(self):
        return f"reach {' >> '.join(str([(k, els[k]) for k in sorted(els.keys())]) for els in self.target_state)}"

    def __call__(self, new_state, environment):
        # if len(self.target_state) == self.ltl_progress:
        #     return 0.0  # the function is done, no influence

        self._update_state(environment)
        # TODO: handle going back in the formula
        if not self.complete():
            active_formula = self.target_state[self.ltl_progress]
            new_ltl_clauses = {k for k, v in active_formula.items() if self.state[k]}
            if not set(active_formula.keys()).difference(new_ltl_clauses):  # TODO: assuming only Trues
                # we are done with this LTL formula, move to the next
                self.ltl_progress += 1
                #     reward = config.option_termination_treshold
                # print(f"Completed LTL step {self.ltl_progress}")
            # elif len(new_ltl_clauses) > len(self.achieved_ltl_clauses):
            #     # Added an extra clause # TODO assuming you can't unset
            #     if self.reward_partials:
            #         print(f"\nwent from {self.achieved_ltl_clauses} \t\t {new_ltl_clauses}")
            #         reward = config.option_trigger_treshold
            #     else:
            #         print("Extra LTL clause, inside option")
            #         reward = 0.1
            self.achieved_ltl_clauses = new_ltl_clauses

        if self.reward_partials:
            reward = len(self.achieved_ltl_clauses) * config.option_trigger_treshold
        else:
            # TODO: multi step state
            reward = len(self.achieved_ltl_clauses) / len(self.target_state[0]) * config.option_trigger_treshold
        return float(reward)

    def _update_state(self, environment):
        grid = environment.env.grid
        self.state['key'] = True  # key is in hand unless it's on the ground
        for idx, cell in enumerate(grid.grid):
            if cell is None:
                continue
            if cell.type == 'wall':
                continue

            if cell.type == 'key':
                self.state['key'] = False
            elif cell.type == 'door':
                self.state['door'] = cell.is_open

            elif cell.type == 'goal':
                y, x = divmod(idx, environment.env.width)
                self.state['goal'] = all(environment.env.agent_pos == (y, x))
            else:
                raise Exception(cell)
        front_cell = environment.env.grid.get(*environment.env.front_pos)
        self.state['front_door'] = False
        if front_cell is not None:
            self.state['front_door'] = front_cell.type == "door"

    def reset(self):
        self.ltl_progress = 0
        self.state = {}
        self.achieved_ltl_clauses = []

    def motivating_function(self, _):
        # TODO: the values should be {True, False, Don't Care}, right now False means Don't Care
        return LTLReward([{k: True for k in self.achieved_ltl_clauses}, ], reward_partials=False)
