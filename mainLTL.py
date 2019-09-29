import random

import config
from main import Search
from rewards import LTLReward


class LTLSearch(Search):
    def __init__(self, population_size):
        super(LTLSearch, self).__init__(population_size)
        self.reward_space_size = 2

    def generate_initial_population(self):
        # population = [LTLReward(
        #     [{'door': random.choice((True, False)),
        #       'goal': random.choice((True, False)),
        #       'front_door': random.choice((True, False))
        #       }],
        # ) for _ in range(config.population)]

        population = [LTLReward(
            [{'door': True,
              'goal': True,
              'front_door': True,
              }],
        ) for _ in range(config.population)]

        print("==INITIAL POPULATION==")
        for r in population:
            print(f"{r.reward_coords}")
        return population

    def test_door(self):
        target_state = LTLReward([
            {'door': True, }
        ])
        door_fitness, _ = self.fitness_function(target_state)
        print('fitness with open doors curiosity', door_fitness, 'door ltl')

    @staticmethod
    def test_door_goal(self):
        target_state = LTLReward([
            {'door': True, 'goal': True},
        ])
        door_fitness, _ = fitness_function(target_state)
        print('fitness step out fitness', door_fitness, 'ltl')


def main():
    config.experiment_name = "LTL"
    config.env_name = "MiniGrid-DoorKey-5x5-v0"
    config.max_nr_options = 3

    regressor = LTLSearch(config.population)
    regressor.optimize()


if __name__ == "__main__":
    main()
