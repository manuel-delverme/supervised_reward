import numpy as np
import cma

class CMAES(cma.CMAEvolutionStrategy):
    """CMA-ES wrapper."""

    def __init__(
            self, num_params,
            sigma_init=0.10,
            popsize=255,
    ):
        self.num_params = num_params
        self.sigma_init = sigma_init
        self.solutions = None
        initial_state = self.num_params * [0]

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
