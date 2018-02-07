import numpy as np
import cma


def compute_ranks(x):
    """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)


# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py

class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.dim = pi.num_params
        self.epsilon = epsilon
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.pi.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class BasicSGD(Optimizer):
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def _compute_step(self, globalg):
        step = -self.stepsize * globalg
        return step


class SGD(Optimizer):
    def __init__(self, pi, stepsize, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


class CMAES(cma.CMAEvolutionStrategy):
    """CMA-ES wrapper."""

    def __init__(self, num_params,  # number of model parameters
                 sigma_init=0.10,  # initial standard deviation
                 popsize=255,  # population size
                 weight_decay=0.01):  # weight decay coefficient

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.weight_decay = weight_decay
        self.solutions = None
        initial_state = self.num_params * [0]
        # initial_state = [0.99, -0.99, 0.99, -0.99]

        super(CMAES, self).__init__(
            initial_state,
            self.sigma_init,
            {
                'popsize': popsize,
                'bounds': [-2, 2],
                'minstd': 0.05,
                # 'integer_variables': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma * sigma))

    def tell(self, solutions, function_values):
        function_values = [-s for s in function_values]
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            function_values += l2_decay
        super(CMAES, self).tell(solutions, function_values)

    def current_param(self):
        return self.result[5]  # mean solution, presumably better with noise

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.result[0]  # best evaluated solution
