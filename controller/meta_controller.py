import numpy as np
import tqdm
import collections
import matplotlib.pyplot as plt
import cma
import multiprocessing
import envs.simple_boxes


class CMAES(object):
    def __init__(self, reward_space_size, population_size, fitness_function, default_args):
        self.population_size = population_size
        self.default_args = default_args
        self.solver = cma.CMAEvolutionStrategy(
            x0=reward_space_size * [0],
            sigma0=0.1,
            inopts={
                'popsize': population_size,
                'bounds': [-0.6, 0.6],
                'minstd': 0.05,
                'CMA_diagonal':  False,
                # 'integer_variables': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            }
        )
        self.fitness_function = fitness_function

    def optimize(self, n_iterations):
        f_h = []
        fbest_h = []
        sigma_h = []

        # fig.show()
        # fig.canvas.draw()
        fig, (fitness_ax, sigma_ax) = plt.subplots(nrows=2)

        SIDE_SIZE = self.default_args[0]
        time_budget = self.default_args[-2]

        import bruteforce_options
        no_options_score = bruteforce_options.get_no_option_score(time_budget)

        f_h_deque = collections.deque(maxlen=5)
        import glob
        try:
            experiment_id = max([int(n[15:-4]) for n in glob.glob("fitness_history*.png")]) + 1
        except:
            experiment_id = 0

        pool = multiprocessing.Pool(processes=self.population_size)
        for optimization_iteration in tqdm.tqdm(range(n_iterations), desc="otimization"):
            solutions_ = self.solver.ask()
            solutions_clip = np.clip(solutions_, -0.6, 0.6)
            solutions = self.scalar_to_coords(SIDE_SIZE, solutions_clip)
            args = ((s,) + self.default_args for s in solutions)

            # async is not needded since **cache is up**, but cache is so slow (gzip?)
            # TODO: with pool as pool
            # fitness_list = pool.map(self.fitness_function, args)
            fitness_list = list(map(self.fitness_function, args))

            results = [-f for f in fitness_list]
            self.solver.tell(solutions_, results)
            fitness_list = np.array(fitness_list)
            f_h_deque.append(fitness_list.mean())
            f_h.append(sum(f_h_deque)/len(f_h_deque))
            fbest_h.append(-self.solver.result.fbest)
            # history.append(self.solver.result.fbest)
            mask = np.ones(shape=(7, 7), dtype=np.str)
            mask[:] = ' '

            for w in envs.simple_boxes.BoxWorldSimple._walls:
                x = w % 7
                y = w // 7
                mask[x, y] = '|'
            best_mask = mask.copy()
            for coords_vec in solutions:
                for g in coords_vec:
                    if g[0] + g[1] * 7 in envs.simple_boxes.BoxWorldSimple._walls:
                        mask[g[0]][g[1]] = '*'
                    else:
                        mask[g[0]][g[1]] = 'X'

            for s in self.scalar_to_coords(SIDE_SIZE, [self.solver.result.xbest]):
                for g in s:
                    if g[0] + g[1] * 7 in envs.simple_boxes.BoxWorldSimple._walls:
                        best_mask[g[0]][g[1]] = '*'
                    else:
                        best_mask[g[0]][g[1]] = 'X'

            pop_fitness = fitness_list[fitness_list > 0].mean()
            sigma_h.append(sum(abs(self.solver.sm.variances))/len(self.solver.sm.variances))
            print("clip:", np.square(solutions_clip - solutions_).sum())
            print("*" * 30)
            print("population: \n{0}\n fitness {1}\n".format(mask.transpose(), np.round(pop_fitness, 2)))
            for s, s_, f in zip(solutions, solutions_, fitness_list):
                print(str(s.ravel()).replace("\n", ""), str((SIDE_SIZE - 1) * (0.5 + s_).ravel()).replace("\n", ""), ":", f)
            print("best: \n{0}\n fitness {1}".format(best_mask.transpose(), np.round(-self.solver.result.fbest, 2)))
            print("sigma", np.round(self.solver.sm.variances, 2))
            print("*" * 30)
            # fitness_ax.cla()
            # fitness_ax.ylim(-100, 100)
            # fitness_ax.set_ylim([-, no_options * 2])
            fitness_ax.plot(fbest_h)
            fitness_ax.plot(f_h)
            # TODO: plot bruteforce optimal
            # TODO: why is no_options not showing? overlap?
            fitness_ax.plot(no_options_score)
            fitness_ax.set_title("fitness")

            # sigma_ax.ylim(0, 100)
            sigma_ax.plot(sigma_h)
            # sigma_ax.set_ylim([0, 20])
            sigma_ax.set_title("sum(sigmas)")

            # Tweak spacing between subplots to prevent labels from overlapping
            plt.subplots_adjust(hspace=0.5)
            # fig.show()
            fig.savefig("fitness_history{}.png".format(experiment_id))
            # fig.canvas.draw()
        pool.close()
        return self.solver.result.xbest, f_h

    def scalar_to_coords(self, SIDE_SIZE, solutions_):
        solutions = [np.clip(SIDE_SIZE * (s + 0.5), 0, SIDE_SIZE - 1).reshape(-1, 2).astype(np.int) for s in solutions_]
        return solutions
