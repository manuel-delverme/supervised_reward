import os
import time

import math
import numpy as np
import tensorboardX


class HyperBand(object):
    def __init__(self, reward_space_size, population_size):
        self.population_size = population_size
        self.reward_space_size = reward_space_size

    def optimize(self, experiment_id, fitness_function, n_iterations):
        log_dir = os.path.join('runs', experiment_id, time.strftime("%Y_%m_%d-%H_%M_%S"))

        run_then_return_val_loss = fitness_function

        # you need to write the following hooks for your custom problem
        def get_random_hyperparameter_configuration():
            return np.random.randn(self.reward_space_size) - 2

        max_iter = 81  # maximum iterations/epochs per configuration
        eta = 3  # defines downsampling rate (default=3)
        logeta = lambda x: math.log(x) / math.log(eta)
        num_successive_havings = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)

        # total number of iterations (w/o reuse) per execution of Successive Halving (n,r)
        iteration_per_halving = (num_successive_havings + 1) * max_iter

        with tensorboardX.SummaryWriter(log_dir, flush_secs=5) as config.tensorboard:

            # Begin Finite Horizon Hyperband outlerloop. Repeat indefinitely.
            for s in reversed(range(num_successive_havings + 1)):

                # Initial number of configurations
                initial_configurations = int(math.ceil(int(iteration_per_halving / max_iter / (s + 1)) * eta ** s))

                r = max_iter * eta ** (-s)  # initial number of iterations to run configurations for

                # Begin Finite Horizon Successive Halving with (n,r)
                T = [get_random_hyperparameter_configuration() for i in range(initial_configurations)]
                for i in range(s + 1):
                    # Run each of the n_i configs for r_i iterations and keep best n_i/eta
                    n_i = initial_configurations * eta ** (-i)
                    r_i = r * eta ** i
                    val_losses = [run_then_return_val_loss(num_iters=r_i, reward_vector=t) for t in T]
                    T = [T[i] for i in np.argsort(val_losses)[0:int(n_i / eta)]]
                # End Finite Horizon Successive Halving with (n,r)
        return None, None

    def eval_solutions(self, fitness_function, solutions):
        # result = pool.map(fitness_function, args)
        result = map(fitness_function, solutions)

        fitness_list, options = list(zip(*result))
        fitness_list = np.array(fitness_list)
        assert not np.isnan(np.min(fitness_list))
        return fitness_list, options
