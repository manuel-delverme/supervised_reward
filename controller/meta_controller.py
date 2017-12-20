import numpy as np
import deap.creator
import deap.base
import deap.tools
import deap.algorithms
import matplotlib.pyplot as plt
import random


class EvolutionaryAlgorithm(object):
    def __init__(self, population_size, fitness_function):
        reward_space_size = 4
        deap.creator.create("FitnessMin", deap.base.Fitness, weights=(1,))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)
        toolbox = deap.base.Toolbox()
        toolbox.register("attribute", random.random)
        toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, toolbox.attribute, n=reward_space_size)
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", deap.tools.cxTwoPoint)
        toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", deap.tools.selTournament, tournsize=3)
        toolbox.register("evaluate", fitness_function)
        self.toolbox = toolbox
        self.population_size = population_size
        # self.population = [[np.random.rand(self.state_size), None] for _ in range(population_size)]

    def optimize(self):
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, logbook = deap.algorithms.eaSimple(
            self.toolbox.population(n=50), self.toolbox, cxpb=0.5, mutpb=0.2, ngen=10,
            stats=stats, halloffame=deap.tools.HallOfFame(1), verbose=True
        )
        gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
        plt.plot(gen, avg, label="average")
        plt.plot(gen, min_, label="minimum")
        plt.plot(gen, max_, label="maximum")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend(loc="lower right")
        plt.show()
