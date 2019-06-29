import numpy as np
import tqdm

import config
import envs.boxes
import envs.hungry_thirsty
import envs.minigrid
import utils.utils
from fitness import fitness_function
from utils import utils


class Search(object):
    def __init__(self, population_size=config.population):
        print("init")
        self.population_size = population_size
        _e = config.environment()
        obs = _e.reset()
        reward_space_size = obs.ravel().shape[0]
        self.reward_space_size = reward_space_size
        print("init done")

    def optimize(self):
        old_fbest = -float('inf')
        random_best = -float('inf')
        old_random_best = -float('inf')

        self.test_intuitive_cases(self.reward_space_size)
        return



        if config.DEBUG:
            _ = fitness_function(self.random.randn(self.reward_space_size))  # raise errors

        population = [np.random.randn(self.reward_space_size) for _ in range(config.population)]
        # fitness = self.get_best_option_score()
        # print(fitness)

        # print('eval baseline: ', end='')
        # baseline, _ = fitness_function(None)
        # print(baseline)
        im_number = 0

        fitness_list = map(fitness_function, population)

        for ind, fit in zip(population, fitness_list):
            utils.plot_intinsic_motivation(np.array(ind), (config.agent_view_size, config.agent_view_size, -1), im_number)
            im_number += 1

            ind.fitness.values = (fit[0],)
            ind.statistics['options'] = fit[1]

        for optimization_iteration in tqdm.tqdm(range(config.evolution_iters), desc="optimization"):

            # solutions = self.solver.ask(number=self.population_size)
            selected_solutions = self._toolbox.select(population)  # , k=len(population))
            population = deap.algorithms.varAnd(selected_solutions, self._toolbox, cxpb, mutpb)

            # fitness_list, options = self.eval_solutions(fitness_function, mdp_parameters, solutions)

            # udpate the new mutated individuals
            invalids = [ind for ind in population if not ind.fitness.valid]
            fitness_list = list(self._toolbox.map(self._toolbox.evaluate, invalids))

            # self.solver.tell(solutions, -fitness_list, copy=True)
            for ind, fit in zip(invalids, fitness_list):
                ind.fitness.values = (fit[0],)
                ind.statistics['options'] = fit[1]
                utils.plot_intinsic_motivation(np.array(ind), (config.agent_view_size, config.agent_view_size, -1), im_number)
                im_number += 1

            hall_of_fame.update(population)

            random_solutions = np.random.randn(self.population_size, self.reward_space_size) - 2
            random_fitness_list, random_options = self.eval_solutions(fitness_function, random_solutions)

            # xbest = self.solver.result.xbest
            # fbest = self.solver.result.fbest

            # xbest = np.array(list(hall_of_fame[0]))
            best_element = hall_of_fame[0]
            fbest = best_element.fitness.values[0]
            best_options = best_element.statistics['options']

            random_best = max(random_best, float(random_fitness_list.max()))
            assert random_best >= old_random_best
            old_random_best = random_best

            stats = self.statistics.compile(population)

            config.tensorboard.add_scalar('optimization/mean', stats['mean'], optimization_iteration)
            config.tensorboard.add_scalar('optimization/min', stats['min'], optimization_iteration)
            config.tensorboard.add_scalar('optimization/max', stats['max'], optimization_iteration)
            config.tensorboard.add_scalar('optimization/std', stats['std'], optimization_iteration)
            config.tensorboard.add_scalar('optimization/best', fbest, optimization_iteration)
            config.tensorboard.add_scalar('optimization/nr_options_best', len(best_options), optimization_iteration)
            # config.tensorboard.add_histogram('optimization/sigmas', self.solver.sm.variances, optimization_iteration)

            config.tensorboard.add_scalar('optimization/random_mean', random_fitness_list.mean(), optimization_iteration)
            config.tensorboard.add_scalar('optimization/random_min', random_fitness_list.min(), optimization_iteration)
            config.tensorboard.add_scalar('optimization/random_max', random_fitness_list.max(), optimization_iteration)
            # config.tensorboard.add_scalar('optimization/random_var', random_fitness_list.var(), optimization_iteration)
            config.tensorboard.add_scalar('optimization/random_best', random_best, optimization_iteration)

            config.tensorboard.add_scalar('optimization/baseline', baseline, optimization_iteration)

            if old_fbest != fbest:
                old_fbest = fbest

        return None, None

    @staticmethod
    def test_intuitive_cases(reward_space_size):
        # door_motivation = np.ones(shape=reward_space_size) * -0.01

        # door_motivation = door_motivation.reshape(config.agent_view_size, config.agent_view_size, -1)
        # door_motivation[:, :, 2] = 1
        # door_fitness = fitness_function(door_motivation.ravel())
        # print('fitness with door curiosity', door_fitness)

        door_motivation = np.ones(shape=reward_space_size) * -0.01
        door_motivation = door_motivation.reshape(config.agent_view_size, config.agent_view_size, -1)

        # for row in range(4):
        # for col in range(4):
        # # 0 1 2 3 4
        # 0 _ _ _ _ _
        # 1 _ _ . _ _
        # 2 _ _ . 1 <
        # 3 _ _ . _ _
        # 4 _ _ _ _ _
        row = 2
        col = 3
        door_motivation[2, 3, 2] = 1
        door_motivation[1:4, 2, 2] = 0.1

        # door_motivation[2, 4, 2] = 1
        better_doors_fitness = fitness_function(door_motivation.ravel())  # raise errors
        print('fitness with front doors curiosity', better_doors_fitness)

    def get_best_option_score(self):
        # options = [learners.approx_q_learning.generate_option(config.environment(), (4, 4, 0), False), ]
        # options = [learners.approx_q_learning.generate_option(config.environment(), (6, 6, 0), False), ]
        print("eval options", str(sorted([str(o) for o in options]))[:50], end=' ')
        fitness = utils.eval_options(envs.minigrid.MiniGrid(), options)
        print(config.option_eval_training_steps, fitness)
        return fitness

    @staticmethod
    def eval_solutions(_fitness_function, solutions):
        # result = pool.map(fitness_function, args)
        result = map(_fitness_function, solutions)

        fitness_list, options = list(zip(*result))
        fitness_list = np.array(fitness_list)
        assert not np.isnan(np.min(fitness_list))
        return fitness_list, options


def main():
    regressor = Search()
    regressor.optimize()


if __name__ == "__main__":
    main()
