import itertools
import numpy as np
import matplotlib.pyplot as plt
import controller.meta_controller
import controller.meta_controller
import envs.simple_boxes
import learners
import options_utils
import functools


# @lru_cache(maxsize=1024)
def fitness_simple_boxes(args):
    coords_vector, SIDE_SIZE, nr_options, possible_box_positions, budget, plot_progress = args
    assert budget % 10 == 0

    goal_idxs = set()
    # print(coords_vector, end=' ')
    # for row in np.nditer(coords_vector, flags=['external_loop'], order='F'):
    for x, y in coords_vector:
        goal_idx = x + y * SIDE_SIZE
        goal_idxs.add(goal_idx)
    # print("ids:", goal_idxs)
    option_set = tuple(sorted(goal_idxs))
    fitnesses, xs = eval_option_set(SIDE_SIZE, option_set)
    # print(fitnesses[int(budget/10)])

    option_names = option_set
    if plot_progress:
        option_names = " ".join(str(n) for n in sorted(option_names))
        plt.plot(xs, fitnesses)
        plt.title(option_names)
        plt.show()
    # print("opt-names:", " ".join(str(n) for n in sorted(option_names)), "fitness:", fitnesses[int(budget / 10)])
    return fitnesses[int(budget / 10)]


@functools.lru_cache(maxsize=1024)
def eval_option_set(SIDE_SIZE, option_set):
    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=SIDE_SIZE)
    xs = [10 + 10 * x for x in range(1000)]
    if token_mdp._walls.intersection(option_set):
        fitnesses = [-100 for x in xs]
    else:
        possible_box_positions = list(
            itertools.combinations([0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE, SIDE_SIZE * SIDE_SIZE - 1, ], 2))
        learner = learners.q_learning.QLearning(env=token_mdp, options=[])
        options = tuple(options_utils.goal_to_policy(learner, goal_idx, token_mdp) for goal_idx in option_set)
        assert tuple(sorted(option_set)) == option_set
        fitnesses = options_utils.eval_options(SIDE_SIZE, options, possible_box_positions, xs)
    return fitnesses, xs


def evolve_coords(nr_options):
    POPULATION_SIZE = 10
    SIDE_SIZE = 7
    plot_progress = False
    time_budget = 100

    reward_space_size = nr_options * 2
    possible_box_positions = tuple(itertools.combinations([0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                                                          SIDE_SIZE * SIDE_SIZE - 1, ], 2))
    regressor = controller.meta_controller.CMAES(
        population_size=POPULATION_SIZE,
        fitness_function=fitness_simple_boxes,
        reward_space_size=reward_space_size,
        default_args=(SIDE_SIZE, nr_options, possible_box_positions, time_budget, plot_progress),
    )
    return regressor.optimize()


def main():
    scores = {
        4: evolve_coords(nr_options=4)
    }
    # for nr_options in range(6):
    #    scores[nr_options] = evolve_weights(nr_options)


if __name__ == "__main__":
    main()
