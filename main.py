import abc
# import multiprocessing
import os
import random
import sys
import time

import numpy as np
import tensorboardX
import tqdm

import config
import fitness
import shared.constants as C
from rewards import Reward, ConstrainedReward

# pool = multiprocessing.Pool()


def maybe_multiprocess(function, population):
    if config.DEBUG or config.visualize_any or not config.multiprocess:
        retr = [function(p) for p in population]
    else:
        global pool
        retr = pool.map(function, population)
    return retr


class Search:
    def __init__(self, population_size=config.population):
        self.population_size = population_size
        _e = config.environment()
        obs = _e.reset()
        reward_space_size = obs.reshape(-1).shape[0]
        self.reward_space_size = reward_space_size

    def fitness_function(self, *args, **kwargs):
        return fitness.fitness_function(*args, **kwargs)

    def optimize(self):
        population = self.generate_initial_population()
        _ = self.fitness_function(population[0])  # raise errors
        raise NotImplementedError

        if config.DEBUG:
            _ = self.fitness_function(population[0])  # raise errors
            self.test_intuitive_cases(self.reward_space_size)

        for optimization_iteration in tqdm.tqdm(range(config.evolution_iters)):
            population_fitness = self.calculate_fitness(population)

            best_sample, best_fitness = max(zip(population, population_fitness), key=lambda x: x[1])
            population = self.mutate(best_sample, population)

            config.tensorboard.add_scalar('optimization/mean', np.array(population_fitness).mean(), optimization_iteration)
            config.tensorboard.add_scalar('optimization/best', best_fitness, optimization_iteration)

        return None, None

    def calculate_fitness(self, population):
        population_fitness, options = zip(*maybe_multiprocess(self.fitness_function, population))

        print("")
        print("==RESULTS==")
        for r, fit in zip(population, population_fitness):
            print(f"{r.reward_coords} achieved fit {fit}")
        return population_fitness

    @staticmethod
    def test_intuitive_cases(reward_space_size):
        # Search.test_door(reward_space_size)
        Search.test_open_door(config.Minigrid.nr_layers * config.agent_view_size * config.agent_view_size)
        # Search.test_door_gradient(reward_space_size)

    @staticmethod
    def eval_solutions(_fitness_function, solutions):
        # result = pool.map(fitness_function, args)
        result = map(_fitness_function, solutions)

        fitness_list, options = list(zip(*result))
        fitness_list = np.array(fitness_list)
        assert not np.isnan(np.min(fitness_list))
        return fitness_list, options

    @abc.abstractmethod
    def generate_initial_population(self):
        pass


class XYLSearch(Search):
    def __init__(self):
        super().__init__()
        self.reward_space_size = 3  # x, y, layer. reward_space_size

    def mutate(self, best_sample, population):
        for idx, p in enumerate(population):
            if p is best_sample:
                continue
            if random.random() > 0.5:
                new_parameters = best_sample.mutate()
                population[idx] = ConstrainedReward(new_parameters, normalize=False)
                print(f'mutate {best_sample.reward_parameters} ({best_sample.reward_coords})',
                      f'to     {population[idx].reward_parameters} ({population[idx].reward_coords})', sep='\n')
        return population

    def generate_initial_population(self):
        population = [ConstrainedReward(np.random.randn(self.reward_space_size)) for _ in range(config.population)]
        # population[0] = ConstrainedReward(np.array([
        #     C.DOOR_LAYER, (config.agent_view_size - 2), (config.agent_view_size // 2)
        # ]), normalize=False)
        print("==INITIAL POPULATION==")
        for r in population:
            print(f"{r.reward_coords}")
        return population

    def test_door_and_goal(self, reward_space_size):
        # D 0 1 2 3 4  G 0 1 2 3 4
        # 0 _ _ _ _ _  0 _ _ _ _ _
        # 1 _ _ . _ _  1 _ _ . _ _
        # 2 _ _ . 1 <  2 _ _ . 1 <
        # 3 _ _ . _ _  3 _ _ . _ _
        # 4 _ _ _ _ _  4 _ _ _ _ _
        intrinsic_motivation = np.ones(shape=reward_space_size) * -0.005
        intrinsic_motivation = intrinsic_motivation.reshape(-1, config.agent_view_size, config.agent_view_size)

        intrinsic_motivation[C.DOOR_LAYER, config.agent_view_size // 2, -2] = 1
        intrinsic_motivation[C.DOOR_LAYER, :, -3] = 0.1

        intrinsic_motivation[C.FOOD_LAYER, config.agent_view_size // 2, -2] = 2
        intrinsic_motivation[C.FOOD_LAYER, :, -3] = 0.2

        door_fitness, _ = self.fitness_function(intrinsic_motivation.reshape(-1))  # raise errors
        print('fitness with door and goal curiosity', door_fitness, intrinsic_motivation.reshape(-1), sep='\n')

    def test_open_door(self, reward_space_size):
        # D 0 1 2 3 4  G 0 1 2 3 4
        # 0 _ _ _ _ _  0 _ _ _ _ _
        # 1 _ _ _ _ _  1 _ _ _ _ _
        # 2 _ _ _ _ _  2 _ _ _ _ _
        # 3 _ _ _ _ _  3 _ _ _ _ _
        # 4 _ 1 * 1 _  4 _ _ _ _ _

        agent_col = config.agent_view_size // 2
        middle = config.agent_view_size // 2

        intrinsic_motivation = np.ones(shape=reward_space_size) * -0.00001
        # intrinsic_motivation = np.ones(shape=reward_space_size) * -0.00001
        intrinsic_motivation = intrinsic_motivation.reshape(-1, config.agent_view_size, config.agent_view_size)

        intrinsic_motivation[C.DOOR_LAYER, -2, agent_col] = 1.1
        # intrinsic_motivation[C.UNWALKABLE_LAYER, -1, agent_col + 1] = 0.6
        # intrinsic_motivation[C.UNWALKABLE_LAYER, -1, agent_col - 1] = 0.6

        reward = Reward(reward_vector=intrinsic_motivation.reshape(-1))
        door_fitness, _ = self.fitness_function(reward)  # raise errors
        print('fitness with open doors curiosity', door_fitness, intrinsic_motivation, sep='\n')

    @staticmethod
    def test_door(reward_space_size):
        # D 0 1 2 3 4  G 0 1 2 3 4
        # 0 _ _ _ _ _  0 _ _ _ _ _
        # 1 _ _ _ _ _  1 _ _ _ _ _
        # 2 _ _ _ _ _  2 _ _ _ _ _
        # 3 _ _ _ _ _  3 _ _ _ _ _
        # 4 _ 1 * 1 _  4 _ _ _ _ _

        agent_col = config.agent_view_size // 2
        middle = config.agent_view_size // 2

        intrinsic_motivation = np.ones(shape=reward_space_size) * -0.0000001
        intrinsic_motivation = intrinsic_motivation.reshape(-1, config.agent_view_size, config.agent_view_size)

        intrinsic_motivation[C.DOOR_LAYER, -2, agent_col] = 0.6

        reward = Reward(reward_vector=intrinsic_motivation.reshape(-1))
        door_fitness, _ = fitness_function(reward)  # raise errors
        print('fitness with open doors curiosity', door_fitness, intrinsic_motivation, sep='\n')


def main():
    experiment_name = "DEBUG:" + time.strftime("%Y_%m_%d-%H_%M_%S")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        import tkinter.simpledialog
    except ImportError:
        HASGUI = False
    else:
        HASGUI = True
    response = None
    if not config.DEBUG and HASGUI:
        try:
            # comment = "256h32bs"
            root = tkinter.Tk()
            response = tkinter.simpledialog.askstring("comment", "comment")
            root.destroy()
        except tkinter.TclError as _:
            pass
        else:
            if response is None:
                response = "DELETEME"
                # DEBUG = True

    config.experiment_name = f'{response}:{time.strftime("%Y_%m_%d-%H_%M_%S")}{os.getpid()}'
    if len(sys.argv) > 1:
        config.experiment_name = f'{sys.argv[1]}:{time.strftime("%Y_%m_%d-%H_%M_%S")}{os.getpid()}'

    print('EXPERIMENT:', experiment_name)
    # class fake_writer:
    #     def add_scalar(*args):
    #         pass
    # tensorboard = fake_writer()
    config.tensorboard = tensorboardX.SummaryWriter(os.path.join('runs', experiment_name), flush_secs=1)

    # multiprocessing.set_start_method('spawn')
    regressor = Search()
    regressor.optimize()


if __name__ == "__main__":
    main()
