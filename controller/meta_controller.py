import numpy as np
import tqdm
import tensorboardX
import matplotlib.pyplot as plt
import cma
import multiprocessing


class CMAES(object):
    def __init__(self, reward_space_size, population_size):
        self.population_size = population_size
        self.reward_space_size = reward_space_size
        self.solver = cma.CMAEvolutionStrategy(
            x0=reward_space_size * [-5.0],
            sigma0=0.5,
            inopts={
                'popsize': population_size,
                # 'bounds': [-0.6, 0.6],
                'minstd': 0.15,
                # 'CMA_diagonal': False,
                # 'integer_variables': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

    def optimize(self, experiment_id, fitness_function, n_iterations, mdp_parameters):
        summary_writer = tensorboardX.SummaryWriter()
        side_size = mdp_parameters['side_size']
        old_fbest = 0

        # with multiprocessing.Pool(processes=self.population_size) as pool:
        if True:
            for optimization_iteration in tqdm.tqdm(range(n_iterations), desc="otimization"):
                solutions = self.solver.ask(number=self.population_size)

                args = []
                for s in solutions:
                    args.append(frozenset({'weights': tuple(s), **mdp_parameters}.items()))
                # async is not needded since **cache is up**, but cache is so slow (gzip?)

                # fitness_list = pool.map(fitness_function, args)
                fitness_list = map(fitness_function, args)
                fitness_list = np.array(list(fitness_list))

                assert not np.isnan(np.min(fitness_list))
                costs = []
                for solution, fitness in zip(solutions, fitness_list):
                #     c = np.clip(solution, -1, np.inf)
                #     regularization = np.sum(c + 1)
                #     cost = -fitness + 0.01 * regularization
                    cost = -fitness
                    costs.append(cost)

                self.solver.tell(solutions, costs)
                fitness_list = np.array(fitness_list)

                summary_writer.add_scalar('optimization/mean', fitness_list.mean(), optimization_iteration)
                summary_writer.add_scalar('optimization/min', fitness_list.min(), optimization_iteration)
                summary_writer.add_scalar('optimization/max', fitness_list.max(), optimization_iteration)
                summary_writer.add_scalar('optimization/var', fitness_list.var(), optimization_iteration)
                summary_writer.add_scalar('optimization/best', -self.solver.result.fbest, optimization_iteration)
                # summary_writer.add_scalar('optimization/baseline', baseline, optimization_iteration)
                summary_writer.add_histogram('optimization/sigmas', self.solver.sm.variances, optimization_iteration)

                xbest = self.solver.result.xbest
                fbest = self.solver.result.fbest

                if old_fbest != fbest:
                    print('UPDATE', fbest)
                    old_fbest = fbest
                    for slice_idx in range(6):
                        slice_from = slice_idx * side_size * side_size
                        slice_to = (slice_idx + 1) * side_size * side_size
                        values = xbest[slice_from:slice_to]
                        value_map = values.reshape(side_size, side_size)
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        cax = ax.matshow(value_map)
                        fig.colorbar(cax)
                        summary_writer.add_figure('best_weights{}'.format(slice_idx), fig, optimization_iteration)


                if optimization_iteration % 10 == 0:
                    summary_writer.file_writer.flush()
                    for path, writer in summary_writer.all_writers.items():
                        writer.flush()


        summary_writer.close()
        return self.solver.result.xbest, None

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
