import random

import config
from main import Search
from rewards import LTLReward
import datetime


class LTLSearch(Search):
    def __init__(self, population_size):
        super(LTLSearch, self).__init__(population_size)
        self.reward_space_size = 2

    def generate_random_sample(self):
        nr_conjunctions = random.randint(2, 4)
        propositions = {
            'door',
            'goal',
            'front_door',
            'key',
            'penguin',
        }
        # unary_operators = [
        #     'past',  # F eventually (finally): true now or sometime in the past (like ◇ of temporal logic)
        #     'global',  # G globally: true now and always (like □ of temporal logic)
        #     'last',  # X next: true in the next state (does not exist in temporal logic)
        #     'until',  # U until: something true until something else happens (does not exist in temporal logic)
        # ]
        binary_operators = [
            # 'until',  # U until: something true until something else happens (does not exist in temporal logic)
            'or',  # implicit for now
            'and',
        ]

        trace = []
        for proposition in random.sample(propositions, nr_conjunctions):
            # operator = random.choice(unary_operators)
            value = random.choice((True, False))
            trace.append((proposition, value))
            trace.append(random.choice(binary_operators))
        del trace[-1]  # remove the last and/or

        trace = [
            ('key', True),
            'and',
            ('front_door', True),
            # 'and',
            # ('door', True),
        ]
        reward = LTLReward(trace)
        return reward

    def generate_initial_population(self):
        population = []
        for _ in range(config.population):
            population.append(self.generate_random_sample())

        print("==INITIAL POPULATION==")
        for r in population:
            print(f"{r.reward_coords}")
        return population


def main():
    config.experiment_name = f"LTL-{datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')}"
    # config.experiment_name = f"LTL"
    config.max_nr_options = 3

    regressor = LTLSearch(config.population)
    regressor.optimize()


if __name__ == "__main__":
    main()
