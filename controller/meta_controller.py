import numpy as np
import deap.creator
import deap.base
import deap.tools
import random


class EvolutionaryAlgorithm(object):
    def __init__(self, env_distr, population_size):
        self.state_size = np.square(env_distr.invariants['size'])
        self.population_size = population_size
        self.population = [[np.random.rand(4) * 2, None] for _ in range(population_size)]

        IND_SIZE = 0
        deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)
        toolbox = deap.base.Toolbox()
        toolbox.register("attribute", random.random)
        toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, toolbox.attribute, n=IND_SIZE)
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            return sum(individual),

        toolbox.register("mate", deap.tools.cxTwoPoint)
        toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", deap.tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)
        self.toolbox = toolbox

    def stuff(self):
        pop = self.toolbox.population(n=50)
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        return pop

    def get_reward_function(self, feature_control=False):
        while True:
            for idx, individual in enumerate(list(self.population)):
                def surrogate_reward(mdp):
                    # TODO: normalize reward function? to remove 1 dof
                    # TODO: add bias?
                    thirst = mdp._state['thirsty']
                    hunger = mdp._state['hungry']
                    reward_vec = individual[0]
                    x = np.array((
                        thirst and hunger,
                        not thirst and not hunger,
                        thirst and not hunger,
                        hunger and not thirst,
                    ), dtype=np.int)
                    # can be optimized as reward_vec[idx]
                    reward = np.dot(reward_vec, x)

                    return reward

                self.population[idx][1] = yield surrogate_reward

            parents = sorted(self.population, key=lambda p: p[1])[:2]
            # prediction = np.random.rand(self.state_size)
            next_generation = list(parents)
            while len(next_generation) < self.population_size:
                offspring = np.empty_like(parents[0][0])
                p0, p1 = [p[0] for p in parents]
                mask = np.random.choice([False, True], size=p0.shape, p=[0.5, 0.5])
                offspring[mask] = p0[mask]
                offspring[~mask] = p1[~mask]
                next_generation.append([offspring, None])
                if random.random() < 0.3:
                    idx = random.randint(0, offspring.shape[0] - 1)
                    offspring[idx] += np.random.normal() / 10
            self.population = list(next_generation)
