import numpy as np
import bruteforce_options
import tqdm
import collections
import matplotlib.pyplot as plt
import cma
import multiprocessing
import envs.simple_boxes


class CMAES(object):
    def __init__(self, reward_space_size, population_size):
        self.population_size = population_size
        self.solver = cma.CMAEvolutionStrategy(
            x0=reward_space_size * [0],
            sigma0=0.1,
            inopts={
                'popsize': population_size,
                'bounds': [-0.6, 0.6],
                'minstd': 0.15,
                'CMA_diagonal': False,
                # 'integer_variables': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

    def optimize(self, experiment_id, fitness_function, n_iterations, mdp_parameters, baseline):
        fitness_means, fitness_bests, cma_es_sigmas = [], [], []
        fig, (fitness_ax, sigma_ax) = plt.subplots(nrows=2)
        fitness_ax.legend(loc='best', shadow=True)
        SIDE_SIZE = mdp_parameters['SIDE_SIZE']

        f_h_deque = collections.deque(maxlen=5)
        with multiprocessing.Pool(processes=self.population_size) as pool:
            for optimization_iteration in tqdm.tqdm(range(n_iterations), desc="otimization"):
                dropped = 0
                legal_solutions = []
                costs = []
                while len(legal_solutions) < self.population_size:
                    normalized_solutions = self.solver.ask(number=self.population_size - len(legal_solutions))
                    scaled_soltuions = self.scalar_to_coords(SIDE_SIZE, normalized_solutions)

                    args = [frozenset({'parameter': tuple(s), **mdp_parameters}.items()) for s in scaled_soltuions]
                    # async is not needded since **cache is up**, but cache is so slow (gzip?)
                    fitness_list = pool.map(fitness_function, args)
                    # fitness_list = list(map(fitness_function, args))

                    for fitness, solution in zip(fitness_list, normalized_solutions):
                        if fitness is not None:
                            legal_solutions.append(solution)
                            costs.append(-fitness)
                        else:
                            dropped += 1

                self.solver.tell(legal_solutions, costs)
                scaled_soltuions = self.scalar_to_coords(SIDE_SIZE, legal_solutions)

                fitness_list = np.array(fitness_list)
                f_h_deque.append(fitness_list.mean())
                fitness_means.append(sum(f_h_deque) / len(f_h_deque))
                fitness_bests.append(-self.solver.result.fbest)
                # history.append(self.solver.result.fbest)
                best_mask, mask = self.draw_mask_ascii(SIDE_SIZE, scaled_soltuions)

                pop_fitness = fitness_list[fitness_list > 0].mean()
                cma_es_sigmas.append(sum(abs(self.solver.sm.variances)) / len(self.solver.sm.variances))

                print("population: \n{0}\n fitness {1}\n".format(mask.transpose(), np.round(pop_fitness, 2)))
                for s, s_, f in zip(scaled_soltuions, normalized_solutions, fitness_list):
                    print(str(s.ravel()).replace("\n", ""), str((SIDE_SIZE - 1) * (0.5 + s_).ravel()).replace("\n", ""),
                          ":", f)
                print("best: \n{0}\n fitness {1}".format(best_mask.transpose(), np.round(-self.solver.result.fbest, 2)))
                print("sigma", np.round(self.solver.sm.variances, 2))
                print("samples rejected for each accepted", float(dropped) / self.population_size)
                print("*" * 30)
                fitness_ax.plot(fitness_bests, label='best')
                fitness_ax.plot(fitness_means, label='mean')
                # TODO: plot bruteforce optimal
                # TODO: why is no_options not showing? overlap?
                fitness_ax.plot([baseline] * len(fitness_means), label='baseline')
                fitness_ax.set_title("fitness")

                sigma_ax.plot(cma_es_sigmas)
                sigma_ax.set_title("sum(sigmas)")

                plt.subplots_adjust(hspace=0.5)
                fig.savefig("fitness_history{}.png".format(experiment_id))
        return self.solver.result.xbest, fitness_means

    def draw_mask_ascii(self, SIDE_SIZE, solutions):
        mask = np.ones(shape=(7, 7), dtype=np.str)
        mask[:] = ' '
        for w in envs.simple_boxes.BoxWorldSimple._walls:
            x = w % 7
            y = w // 7
            mask[x, y] = '|'
        best_mask = mask.copy()
        for coords_vec in solutions:
            for g in coords_vec.reshape(-1, 2):
                if g[0] + g[1] * 7 in envs.simple_boxes.BoxWorldSimple._walls:
                    mask[g[0]][g[1]] = '*'
                else:
                    mask[g[0]][g[1]] = 'X'
        for s in self.scalar_to_coords(SIDE_SIZE, [self.solver.result.xbest]):
            for g in np.array(s).reshape(-1, 2):
                if g[0] + g[1] * 7 in envs.simple_boxes.BoxWorldSimple._walls:
                    best_mask[g[0]][g[1]] = '*'
                else:
                    best_mask[g[0]][g[1]] = 'X'
        return best_mask, mask

    def scalar_to_coords(self, SIDE_SIZE, solutions_):
        solutions = [np.clip(SIDE_SIZE * (s + 0.5), 0, SIDE_SIZE - 1).astype(np.int) for s in solutions_]
        return solutions
