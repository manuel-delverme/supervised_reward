import numpy as np
import tqdm

import config
import shared.utils
from fitness import fitness_function


class LTLReward:
    def __init__(self, target_state):
        self.target_state = target_state
        self.state = {}
        self.ltl_progress = 0

    def __repr__(self):
        return f"reach {' >> '.join(str([(k, els[k]) for k in sorted(els.keys())]) for els in self.target_state)}"

    def __call__(self, new_state, environment):
        grid = environment.env.grid
        achieved_states = sum(self.state.values())

        for cell in grid.grid:
            if cell is None:
                continue

            if cell.type == 'door':
                self.state['door'] = cell.is_open

            if cell.type == 'goal':
                self.state['goal'] = all(cell.cur_pos == environment.agent_position_idx)

        # what about:
        # open door > step out > close door > reach goal?
        completition = []
        for key, value in self.target_state[self.ltl_progress].items():
            completition.append(self.state[key] == value)

        if all(completition):
            self.ltl_progress += 1
            if len(self.target_state) == self.ltl_progress:
                return config.option_termination_treshold
            return 0.1

        elif sum(self.state.values()) > achieved_states:
            return config.option_trigger_treshold
        else:
            return -0.1

    def reset(self):
        self.ltl_progress = 0
        self.state = {}

    def motivating_function(self):
        return LTLReward([self.state.copy()])


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
        Search.test_door_goal()

    @staticmethod
    def test_door():
        target_state = LTLReward([
            {'door': True, }
        ])
        door_fitness, _ = fitness_function(target_state)
        print('fitness with open doors curiosity', door_fitness, 'door ltl')

    @staticmethod
    def test_door_goal():
        target_state = LTLReward([
            {'door': True, 'goal': True},
        ])
        door_fitness, _ = fitness_function(target_state)
        print('fitness step out fitness', door_fitness, 'ltl')

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
