import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cma
import es

import random


class CMAES(object):
    def __init__(self, reward_space_size, population_size, fitness_function):
        self.solver = cma.CMAEvolutionStrategy(
            x0=reward_space_size * [0],
            sigma0=0.1,
            inopts={
                'popsize': population_size,
                'bounds': [-2, 2],
                'minstd': 0.05,
                # 'integer_variables': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            }
        )
        self.fitness_function = fitness_function

    def hand_optimize(self):
        with open("/tmp/fitnesses.log", "w") as fout:
            pass

        lambda0 = 4
        mean = None
        step_size = None
        covariance = np.identity(5)
        evolution_path_sigma = 0
        evolution_path_c = 0

        while True:
            results = []
            for i in range(1, lambda0):
                xi = np.random.multivariate_normal(mean=mean, cov=np.power(step_size, 2) * covariance)
                fitness = self.fitness_function(xi)
                results.append((xi, fitness))

            # can be skipped by heap
            results = sorted(results, key=lambda x: x[1])

            weights = np.array([1, 2, 3, 4, 5])
            weights /= weights.sum()

            x = [xi for xi, f in results]
            m_prime = mean
            # mean =

            # evolution_path_sigma = update_psigma
            # evolution_path_c = update_pc
            # covariance = update_C
            # step_size = update_sigma

            solutions = self.solver.ask()
            fitness_list = []
            for solution in solutions:
                fitness_list.append(fitness)
                with open("/tmp/fitnesses.log", "a") as fout:
                    fout.write(str(fitness) + "\n")
            print(fitness_list)
            self.solver.tell(solutions, fitness_list)
            # history.append(self.solver.result.fbest)
            print("best", str(self.solver.result.xbest).replace("\n", " "), "fitness", self.solver.result.fbest)
            print("sigma", str(self.solver.sigma))
            # cma.plot()  # shortcut for es.logger.plot()
        return history

    def optimize(self):
        f_h = []

        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()

        while True:
            solutions = self.solver.ask()
            fitness_list = []
            # bins = np.array([-0.99, -0.5, 0.0, 0.5, 0.99]) - 0.25
            # inds = [bins[idx - 1] + 0.25 for idx in np.digitize(solutions, bins)]
            # solutions = inds
            for solution in solutions:
                fitness = self.fitness_function(solution)
                fitness_list.append(fitness)
            # print(fitness_list)
            self.solver.tell(solutions, fitness_list)
            f_h.append(np.array(fitness_list).mean())
            # history.append(self.solver.result.fbest)
            print("*" * 30)
            print("ran", len(solutions), "solutions, scores:", [int(f) for f in fitness_list])
            print("best {0} fitness {1}".format(np.round(self.solver.result.xbest.reshape(3,3), 2), self.solver.result.fbest))
            print("sigma", str(self.solver.sigma))
            print("*" * 30)
            plt.plot(f_h)
            plt.savefig("fitness_history.png")
            # fig.canvas.draw()

        return history
