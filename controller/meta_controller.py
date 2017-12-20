import numpy as np
import tqdm
import deap.creator
import deap.base
import deap.tools
import deap.algorithms
import matplotlib.pyplot as plt
import random


class EvolutionaryAlgorithm(object):
    def __init__(self, population_size, fitness_function):
        IND_SIZE = 4
        deap.creator.create("FitnessMin", deap.base.Fitness, weights=(1, ))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)
        toolbox = deap.base.Toolbox()
        toolbox.register("attribute", random.random)
        toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, toolbox.attribute, n=IND_SIZE)
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

        def evaluate(individual):
            def surrogate_reward(mdp):
                # TODO: normalize reward function? to remove 1 dof
                # TODO: add bias?
                reward_vec = individual[0]
                # state = np.zeros_like(reward_vec)
                # state[mdp.state] = 1
                # for state_idx in mdp.terminal_states:
                #     state[mdp.state] = -1
                # return np.dot(mdp.state, individual[0])
                # special case for this gridworld?
                reward = reward_vec[mdp.agent_position_idx]
                return reward
            return (yield surrogate_reward)

        toolbox.register("mate", deap.tools.cxTwoPoint)
        toolbox.register("mutate", deap.tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", deap.tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)
        self.toolbox = toolbox
        self.state_size = np.square(env_distr.invariants['size'])
        self.population_size = population_size
        # self.population = [[np.random.rand(self.state_size), None] for _ in range(population_size)]

        pop = self.toolbox.population(n=50)
        hof = deap.tools.HallOfFame(1)
        stats = deap.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        pop, logbook = deap.algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)
        gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
        plt.plot(gen, avg, label="average")
        plt.plot(gen, min_, label="minimum")
        plt.plot(gen, max_, label="maximum")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend(loc="lower right")
        plt.show()
        return pop

    def get_reward_function(self, feature_control=False):
        while True:
            for idx, individual in enumerate(list(self.population)):
                def surrogate_reward(mdp):
                    # TODO: normalize reward function? to remove 1 dof
                    # TODO: add bias?
                    reward_vec = individual[0]
                    # state = np.zeros_like(reward_vec)
                    # state[mdp.state] = 1
                    # for state_idx in mdp.terminal_states:
                    #     state[mdp.state] = -1
                    # return np.dot(mdp.state, individual[0])
                    # special case for this gridworld?
                    reward = reward_vec[mdp.agent_position_idx]
                    return reward

                self.population[idx][1] = yield surrogate_reward

            parents = sorted(self.population, key=lambda p: p[1])[:2]
            # prediction = np.random.rand(self.state_size)
            next_generation = []
            while len(next_generation) < self.population_size:
                offspring = np.empty(self.state_size)
                p0, p1 = [p[0] for p in parents]
                mask = np.random.choice([False, True], size=p1.shape, p=[0.5, 0.5])
                offspring[mask] = p0[mask]
                offspring[~mask] = p1[~mask]
                next_generation.append(offspring)
