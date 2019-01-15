import numpy as np
import shutil
import os
import tqdm
import tensorboardX
import matplotlib.pyplot as plt
import cma


class CMAES(object):
    def __init__(self, reward_space_size, population_size):
        self.population_size = population_size
        self.reward_space_size = reward_space_size
        self.solver = cma.CMAEvolutionStrategy(
            x0=reward_space_size * [-2.0],
            sigma0=0.5,
            inopts={
                'popsize': population_size,
                'minstd': 0.15,
            }
        )

    def optimize(self, experiment_id, fitness_function, n_iterations, mdp_parameters):
        log_dir = os.path.join('runs', experiment_id)
        try:
            shutil.rmtree(log_dir, ignore_errors=False)
        except FileNotFoundError:
            pass
        with tensorboardX.SummaryWriter(log_dir, flush_secs=5) as summary_writer:
            layout = {
                'best': {'optimization': ['Multiline',
                                          ['optimization/baseline', 'optimization/best', 'optimization/random_best']]},
                'mean': {'optimization': ['Multiline',
                                          ['optimization/baseline', 'optimization/mean', 'optimization/random_mean']]},
            }
            summary_writer.add_custom_scalars(layout)

            side_size = mdp_parameters['side_size']
            old_fbest = 0
            random_best = 0.0
            old_random_best = -1.0
            baseline, _ = fitness_function((frozenset({'weights': None, **mdp_parameters}.items())))

            for optimization_iteration in tqdm.tqdm(range(n_iterations), desc="optimization"):

                solutions = self.solver.ask(number=self.population_size)
                fitness_list, options = self.eval_solutions(fitness_function, mdp_parameters, solutions)
                self.solver.tell(solutions, -fitness_list, copy=True)

                random_solutions = np.random.randn(self.population_size, self.reward_space_size) - 2
                random_fitness_list, random_options = self.eval_solutions(fitness_function, mdp_parameters,
                                                                          random_solutions)

                xbest = self.solver.result.xbest
                fbest = self.solver.result.fbest

                random_best = max(random_best, float(random_fitness_list.max()))
                assert random_best >= old_random_best
                old_random_best = random_best

                summary_writer.add_scalar('optimization/mean', fitness_list.mean(), optimization_iteration)
                summary_writer.add_scalar('optimization/min', fitness_list.min(), optimization_iteration)
                summary_writer.add_scalar('optimization/max', fitness_list.max(), optimization_iteration)
                summary_writer.add_scalar('optimization/var', fitness_list.var(), optimization_iteration)
                summary_writer.add_scalar('optimization/best', -self.solver.result.fbest, optimization_iteration)
                summary_writer.add_histogram('optimization/sigmas', self.solver.sm.variances, optimization_iteration)

                summary_writer.add_scalar('optimization/random_mean', random_fitness_list.mean(),
                                          optimization_iteration)
                summary_writer.add_scalar('optimization/random_min', random_fitness_list.min(), optimization_iteration)
                summary_writer.add_scalar('optimization/random_max', random_fitness_list.max(), optimization_iteration)
                summary_writer.add_scalar('optimization/random_var', random_fitness_list.var(), optimization_iteration)
                summary_writer.add_scalar('optimization/random_best', random_best, optimization_iteration)

                summary_writer.add_scalar('optimization/baseline', baseline, optimization_iteration)

                if old_fbest != fbest:
                    old_fbest = fbest
                    for idx, values in enumerate(xbest.reshape(-1, side_size * side_size)):
                        value_map = values.reshape(side_size, side_size)
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        cax = ax.matshow(value_map)
                        fig.colorbar(cax)
                        summary_writer.add_figure('best_weights{}'.format(idx), fig, optimization_iteration)

                    opts = options[np.argwhere(fitness_list == -fbest)[0][0]]
                    opt_map = np.zeros((side_size, side_size))
                    for opt in opts:
                        for idx, action in enumerate(opt):
                            if action == -1:
                                goal_pos = idx % (side_size * side_size)
                                y, x = divmod(goal_pos, side_size)
                                opt_map[y][x] = 1
                                break
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    cax = ax.matshow(opt_map)
                    fig.colorbar(cax)
                    summary_writer.add_figure('best_options', fig, optimization_iteration)

        return self.solver.result.xbest, None

    def eval_solutions(self, fitness_function, mdp_parameters, solutions):
        args = []
        for s in solutions:
            args.append(frozenset({'weights': tuple(s), **mdp_parameters}.items()))
        # result = pool.map(fitness_function, args)
        result = map(fitness_function, args)
        fitness_list, options = list(zip(*result))
        fitness_list = np.array(fitness_list)
        assert not np.isnan(np.min(fitness_list))
        fitness_list = np.array(fitness_list)
        return fitness_list, options
