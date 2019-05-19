import operator

import deap.algorithms
import deap.base
import deap.creator
import deap.tools
import gym
import numpy as np
import tqdm

import config
import envs.boxes
import envs.hungry_thirsty
import envs.minigrid
import learners.approx_q_learning
from utils import utils


def fitness_function(reward_vector):
    if reward_vector is None:
        intrinsic_reward_function = None
        generate_options = False
    else:
        reward_vector = np.array(reward_vector)
        generate_options = True

        def intrinsic_reward_function(observation, info):
            image = observation.ravel()
            return reward_vector[:-1].dot(image) + reward_vector[-1]

    # draw training environment
    mdp = config.environment()

    if intrinsic_reward_function is None:
        options = []
    else:
        # generate options in that
        options, _, _, _ = learners.approx_q_learning.learn(environment=mdp, surrogate_reward=intrinsic_reward_function, training_steps=config.option_discovery_steps,
                                                            generate_options=generate_options, eval_fitness=False)

    print("eval options", str(sorted([str(o) for o in options]))[:100], end=' ')
    fitness = utils.eval_options(envs.minigrid.MiniGrid(), options)
    print(fitness)
    return fitness, options


class GeneticEvolution(object):
    def __init__(self, population_size=config.population):
        print("init")
        self.population_size = population_size

        reward_space_size = config.environment().reset().ravel().shape[0] + 1

        deap.creator.create('FitnessMax', deap.base.Fitness, weights=(1.0,))
        deap.creator.create('Individual', list, fitness=deap.creator.FitnessMax, statistics=dict)

        self._toolbox = deap.base.Toolbox()
        # self._toolbox.register('expr', self._gen_grow_safe, pset=self._pset, min_=1, max_=3)
        # self._toolbox.register('individual', , deap.creator.Individual, self._toolbox.expr)
        # self._toolbox.register('compile', self._compile_to_sklearn)
        self._toolbox.register('mate', deap.tools.cxTwoPoint)
        # self._toolbox.register('expr_mut', self._gen_grow_safe, min_=1, max_=4)
        self._toolbox.register('mutate', deap.tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.2)
        # self._toolbox.register('mutate', deap.tools.mutShuffleIndexes, indpb=0.2)

        # pool = multiprocessing.Pool(processes=6)
        # self._toolbox.register("map", pool.map)
        self._toolbox.register("map", map)

        self._toolbox.register("attr_init", lambda: (np.random.randn() - (2 / reward_space_size)))
        self._toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, self._toolbox.attr_init, n=reward_space_size)
        # self._toolbox.register("select", deap.tools.selBest, k=3)
        self._toolbox.register("select", deap.tools.selTournament, k=population_size, tournsize=3)
        self._toolbox.register('population', deap.tools.initRepeat, list, self._toolbox.individual)
        self.statistics = deap.tools.Statistics(key=operator.attrgetter("fitness.values"))

        self.statistics.register("max", np.max)
        self.statistics.register("mean", np.mean)
        self.statistics.register("min", np.min)
        self.statistics.register("std", np.std)
        self.reward_space_size = reward_space_size
        print("init done")

    def optimize(self, experiment_id):
        self._toolbox.register("evaluate", fitness_function)
        hall_of_fame = deap.tools.HallOfFame(maxsize=10)
        old_fbest = -float('inf')
        random_best = -float('inf')
        old_random_best = -float('inf')

        cxpb, mutpb = 0.5, 0.2
        population = self._toolbox.population(n=self.population_size)

        if config.DEBUG:
            _ = fitness_function(population[0])  # raise errors

        # fitness = self.get_best_option_score()
        # print(fitness)

        print('eval baseline: ', end='')
        baseline, _ = fitness_function(None)
        print(baseline)
        raise Exception

        fitness_list = self._toolbox.map(self._toolbox.evaluate, population)
        for ind, fit in zip(population, fitness_list):
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

    def get_best_option_score(self):
        # options = [learners.approx_q_learning.generate_option(config.environment(), (4, 4, 0), False), ]
        # options = [learners.approx_q_learning.generate_option(config.environment(), (6, 6, 0), False), ]
        print("eval options", str(sorted([str(o) for o in options]))[:50], end=' ')
        fitness = utils.eval_options(envs.minigrid.MiniGrid(), options)
        print(config.option_eval_training_steps, fitness)
        return fitness

    def eval_solutions(self, fitness_function, solutions):
        # result = pool.map(fitness_function, args)
        result = map(fitness_function, solutions)

        fitness_list, options = list(zip(*result))
        fitness_list = np.array(fitness_list)
        assert not np.isnan(np.min(fitness_list))
        return fitness_list, options
