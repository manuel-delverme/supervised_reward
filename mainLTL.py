import numpy as np
import tqdm

import config
import envs.boxes
import envs.hungry_thirsty
import envs.minigrid
import shared.constants as C
import shared.utils
from fitness import fitness_function


class LTLReward:
    def __init__(self, target_state):
        self.target_state = target_state
        self.state = {}

    def __repr__(self):
        return f"reach {self.target_state}"

    def __call__(self, new_state, environment):
        grid = environment.env.grid
        for cell in grid.grid:
            if cell is None:
                continue

            # def progress_condition(_cell):
            #     return _cell.type == 'door' and _cell.is_open == self.door
            # if progress_condition(cell):

            if cell.type == 'door':
                self.state['door'] = cell.is_open

                if self.state['door'] == self.target_state['door']:
                    return 1.0
        else:
            return -0.1

    def motivating_function(self):
        return LTLReward(self.state.copy())


class Search(object):
    def __init__(self, population_size=config.population):
        print("init")
        self.population_size = population_size
        _e = config.environment()
        obs = _e.reset()
        reward_space_size = obs.reshape(-1).shape[0]
        self.reward_space_size = reward_space_size
        print("init done")

    def optimize(self):
        old_fbest = -float('inf')
        random_best = -float('inf')
        old_random_best = -float('inf')

        self.test_intuitive_cases(self.reward_space_size)
        return

        if config.DEBUG:
            _ = fitness_function(self.random.randn(self.reward_space_size))  # raise errors

        population = [np.random.randn(self.reward_space_size) for _ in range(config.population)]
        # fitness = self.get_best_option_score()
        # print(fitness)

        # print('eval baseline: ', end='')
        # baseline, _ = fitness_function(None)
        # print(baseline)
        im_number = 0

        fitness_list = map(fitness_function, population)

        for ind, fit in zip(population, fitness_list):
            shared.plot_intinsic_motivation(np.array(ind), (-1, config.agent_view_size, config.agent_view_size), im_number)
            im_number += 1

            ind.fitness.values = (fit[0],)
            ind.statistics['options'] = fit[1]

        for optimization_iteration in tqdm.tqdm(range(config.evolution_iters), desc="optimization"):

            # solutions = self.solver.ask(number=self.population_size)
            selected_solutions = self._toolbox.select(population)  # , k=len(population))
            population = deap.algorithms.varAnd(selected_solutions, self._toolbox, cxpb, mutpb)

            # fitness_list, options = self.eval_solutions(fitness_function, mdp_parameters, solutions)

            # udpate the new mutated individuals
            invalids = [ind for ind in population if not ind.fitness.valid]
            fitness_list = list(self._toolbox.map(self._toolbox.evaluate, invalids))

            # self.solver.tell(solutions, -fitness_list, copy=True)
            for ind, fit in zip(invalids, fitness_list):
                ind.fitness.values = (fit[0],)
                ind.statistics['options'] = fit[1]
                shared.plot_intinsic_motivation(np.array(ind), (-1, config.agent_view_size, config.agent_view_size), im_number)
                im_number += 1

            hall_of_fame.update(population)

            random_solutions = np.random.randn(self.population_size, self.reward_space_size) - 2
            random_fitness_list, random_options = self.eval_solutions(fitness_function, random_solutions)

            # xbest = self.solver.result.xbest
            # fbest = self.solver.result.fbest

            # xbest = np.array(list(hall_of_fame[0]))
            best_element = hall_of_fame[0]
            fbest = best_element.fitness.values[0]
            best_options = best_element.statistics['options']

            random_best = max(random_best, float(random_fitness_list.max()))
            assert random_best >= old_random_best
            old_random_best = random_best

            stats = self.statistics.compile(population)

            config.tensorboard.add_scalar('optimization/mean', stats['mean'], optimization_iteration)
            config.tensorboard.add_scalar('optimization/min', stats['min'], optimization_iteration)
            config.tensorboard.add_scalar('optimization/max', stats['max'], optimization_iteration)
            config.tensorboard.add_scalar('optimization/std', stats['std'], optimization_iteration)
            config.tensorboard.add_scalar('optimization/best', fbest, optimization_iteration)
            config.tensorboard.add_scalar('optimization/nr_options_best', len(best_options), optimization_iteration)
            # config.tensorboard.add_histogram('optimization/sigmas', self.solver.sm.variances, optimization_iteration)

            config.tensorboard.add_scalar('optimization/random_mean', random_fitness_list.mean(), optimization_iteration)
            config.tensorboard.add_scalar('optimization/random_min', random_fitness_list.min(), optimization_iteration)
            config.tensorboard.add_scalar('optimization/random_max', random_fitness_list.max(), optimization_iteration)
            # config.tensorboard.add_scalar('optimization/random_var', random_fitness_list.var(), optimization_iteration)
            config.tensorboard.add_scalar('optimization/random_best', random_best, optimization_iteration)

            config.tensorboard.add_scalar('optimization/baseline', baseline, optimization_iteration)

            if old_fbest != fbest:
                old_fbest = fbest

        return None, None

    @staticmethod
    def test_intuitive_cases(reward_space_size):
        Search.test_open_door(reward_space_size)

    @staticmethod
    def test_door_and_goal(reward_space_size):
        # D 0 1 2 3 4  G 0 1 2 3 4
        # 0 _ _ _ _ _  0 _ _ _ _ _
        # 1 _ _ . _ _  1 _ _ . _ _
        # 2 _ _ . 1 <  2 _ _ . 1 <
        # 3 _ _ . _ _  3 _ _ . _ _
        # 4 _ _ _ _ _  4 _ _ _ _ _
        intrinsic_motivation = np.ones(shape=reward_space_size) * -0.005
        intrinsic_motivation = intrinsic_motivation.reshape(-1, config.agent_view_size, config.agent_view_size)

        intrinsic_motivation[C.DOOR_LAYER, config.agent_view_size // 2, -2] = 1
        intrinsic_motivation[C.DOOR_LAYER, :, -3] = 0.1

        intrinsic_motivation[C.FOOD_LAYER, config.agent_view_size // 2, -2] = 2
        intrinsic_motivation[C.FOOD_LAYER, :, -3] = 0.2

        door_fitness, _ = fitness_function(intrinsic_motivation.reshape(-1))  # raise errors
        print('fitness with door and goal curiosity', door_fitness, intrinsic_motivation.reshape(-1), sep='\n')

    @staticmethod
    def test_open_door(reward_space_size):

        door_ltl = LTLReward({
            'door': True,
        })
        door_fitness, _ = fitness_function(door_ltl)  # raise errors
        print('fitness with open doors curiosity', door_fitness, 'door ltl')

    @staticmethod
    def test_door_gradient(reward_space_size):
        agent_row = config.agent_view_size // 2
        middle = config.agent_view_size // 2

        intrinsic_motivation = np.ones(shape=reward_space_size) * -0.01
        intrinsic_motivation = intrinsic_motivation.reshape(-1, config.agent_view_size, config.agent_view_size)

        intrinsic_motivation[C.DOOR_LAYER, agent_row, -2] = 1.0
        intrinsic_motivation[C.UNWALKABLE_LAYER, agent_row - 1, -3] = 0.01
        intrinsic_motivation[C.UNWALKABLE_LAYER, agent_row + 1, -3] = 0.01
        door_fitness, _ = fitness_function(intrinsic_motivation.reshape(-1))  # raise errors
        print('fitness with front doors curiosity', door_fitness, intrinsic_motivation, sep='\n')

    @staticmethod
    def test_door(reward_space_size):
        # D 0 1 2 3 4  G 0 1 2 3 4
        # 0 _ _ _ _ _  0 _ _ _ _ _
        # 1 _ _ . _ _  1 _ _ . _ _
        # 2 _ _ . 1 <  2 _ _ . 1 <
        # 3 _ _ . _ _  3 _ _ . _ _
        # 4 _ _ _ _ _  4 _ _ _ _ _

        agent_row = config.agent_view_size // 2
        middle = config.agent_view_size // 2

        intrinsic_motivation = np.ones(shape=reward_space_size) * -0.005
        intrinsic_motivation = intrinsic_motivation.reshape(-1, config.agent_view_size, config.agent_view_size)
        intrinsic_motivation[C.DOOR_LAYER, agent_row, -2] = 1.0

        door_fitness, _ = fitness_function(intrinsic_motivation.reshape(-1))  # raise errors
        print('fitness with front doors curiosity', door_fitness, intrinsic_motivation, sep='\n')

    def get_best_option_score(self):
        # options = [learners.approx_q_learning.generate_option(config.environment(), (4, 4, 0), False), ]
        # options = [learners.approx_q_learning.generate_option(config.environment(), (6, 6, 0), False), ]
        print("eval options", str(sorted([str(o) for o in options]))[:50], end=' ')
        fitness = shared.eval_options(envs.minigrid.MiniGrid(), options)
        print(config.option_eval_training_steps, fitness)
        return fitness

    @staticmethod
    def eval_solutions(_fitness_function, solutions):
        # result = pool.map(fitness_function, args)
        result = map(_fitness_function, solutions)

        fitness_list, options = list(zip(*result))
        fitness_list = np.array(fitness_list)
        assert not np.isnan(np.min(fitness_list))
        return fitness_list, options


def main():
    regressor = Search()
    regressor.optimize()


if __name__ == "__main__":
    main()
