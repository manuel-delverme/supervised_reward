import numpy as np

import cma
import es

import random


class EvolutionaryAlgorithm(object):
    def __init__(self, population_size, fitness_function, reward_space_size):
        import deap.algorithms
        import deap.creator
        import deap.base
        import deap.tools
        deap.creator.create("FitnessMin", deap.base.Fitness, weights=(1,))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)
        toolbox = deap.base.Toolbox()
        toolbox.register("attribute", lambda: random.random() - 0.5)
        toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, toolbox.attribute,
                         n=reward_space_size)
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", deap.tools.cxTwoPoint)
        toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", deap.tools.selTournament, tournsize=3)
        toolbox.register("evaluate", fitness_function)
        self.toolbox = toolbox
        self.population_size = population_size
        # self.population = [[np.random.rand(self.state_size), None] for _ in range(population_size)]

    def optimize(self):
        import deap.tools
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, logbook = deap.algorithms.eaSimple(
            self.toolbox.population(n=10), self.toolbox, cxpb=0.5, mutpb=0.2, ngen=10,
            stats=stats, halloffame=deap.tools.HallOfFame(1), verbose=True
        )
        gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
        import matplotlib.pyplot as plt
        plt.plot(gen, avg, label="average")
        plt.plot(gen, min_, label="minimum")
        plt.plot(gen, max_, label="maximum")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend(loc="lower right")
        plt.show()


class CMAES(object):
    def __init__(self, reward_space_size, population_size, fitness_function, weight_decay=0.0, sigma_init=0.5):
        self.solver = es.CMAES(
            reward_space_size,
            popsize=population_size,
            weight_decay=0.0,
            sigma_init=0.5
        )
        self.fitness_function = fitness_function

    def optimize(self):
        history = []
        while True:
            solutions = self.solver.ask()
            fitness_list = []
            for solution in solutions:
                fitness = self.fitness_function(solution)
                fitness_list.append(fitness)
            print(fitness_list)
            self.solver.tell(solutions, fitness_list)
            # history.append(self.solver.result.fbest)
            print("best", str(self.solver.result.xbest).replace("\n", " "), "fitness", self.solver.result.fbest)
            # cma.plot()  # shortcut for es.logger.plot()
        return history
