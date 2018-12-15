import numpy as np
import tqdm
import tensorboardX
import matplotlib.pyplot as plt
import cma


class CMAES(object):
    def __init__(self, reward_space_size, population_size):
        self.population_size = population_size
        self.reward_space_size = reward_space_size
        self.solver = cma.CMAEvolutionStrategy(
            x0=reward_space_size * [-5.0],
            sigma0=0.5,
            inopts={
                'popsize': population_size,
                'minstd': 0.15,
            }
        )

    def optimize(self, experiment_id, fitness_function, n_iterations, mdp_parameters):
        summary_writer = tensorboardX.SummaryWriter(experiment_id)
        layout = {
            'best': {'optimization': ['Multiline', ['optimization/baseline', 'optimization/best']]},
            'mean': {'optimization': ['Multiline', ['optimization/baseline', 'optimization/mean']]},
        }
        summary_writer.add_custom_scalars(layout)

        side_size = mdp_parameters['side_size']
        old_fbest = 0
        baseline = fitness_function((frozenset({'weights': None, **mdp_parameters}.items())))

        for optimization_iteration in tqdm.tqdm(range(n_iterations), desc="otimization"):
            solutions = self.solver.ask(number=self.population_size)

            args = []
            for s in solutions:
                args.append(frozenset({'weights': tuple(s), **mdp_parameters}.items()))

            # fitness_list = pool.map(fitness_function, args)
            fitness_list = map(fitness_function, args)
            fitness_list = np.array(list(fitness_list))

            assert not np.isnan(np.min(fitness_list))
            costs = []
            for solution, fitness in zip(solutions, fitness_list):
                cost = -fitness
                costs.append(cost)

            self.solver.tell(solutions, costs)
            fitness_list = np.array(fitness_list)

            summary_writer.add_scalar('optimization/mean', fitness_list.mean(), optimization_iteration)
            summary_writer.add_scalar('optimization/min', fitness_list.min(), optimization_iteration)
            summary_writer.add_scalar('optimization/max', fitness_list.max(), optimization_iteration)
            summary_writer.add_scalar('optimization/var', fitness_list.var(), optimization_iteration)
            summary_writer.add_scalar('optimization/best', -self.solver.result.fbest, optimization_iteration)
            summary_writer.add_scalar('optimization/baseline', baseline, optimization_iteration)
            summary_writer.add_histogram('optimization/sigmas', self.solver.sm.variances, optimization_iteration)

            xbest = self.solver.result.xbest
            fbest = self.solver.result.fbest

            if old_fbest != fbest:
                old_fbest = fbest
                for idx, values in enumerate(xbest.reshape(-1, side_size * side_size)):
                    value_map = values.reshape(side_size, side_size)
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    cax = ax.matshow(value_map)
                    fig.colorbar(cax)
                    summary_writer.add_figure('best_weights{}'.format(idx), fig, optimization_iteration)

            if optimization_iteration % 10 == 0:
                summary_writer.file_writer.flush()
                for path, writer in summary_writer.all_writers.items():
                    writer.flush()

        summary_writer.close()
        return self.solver.result.xbest, None
