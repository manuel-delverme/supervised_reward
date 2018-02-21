import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import envs.simple_boxes
import learners
import controller.meta_controller
import itertools
import numpy as np
import controller.meta_controller
import envs.simple_boxes as e
import options_utils


def fitness_simple_boxes(args):
    coords_vector, SIDE_SIZE, nr_options, possible_box_positions, budget, plot_progress = args
    assert budget % 10 == 0

    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=SIDE_SIZE)

    goal_idxs = set()
    for x, y in list(zip(coords_vector, coords_vector[1:]))[::2]:
        x = max(0, x)
        y = max(0, y)
        x = min(SIDE_SIZE, x)
        y = min(SIDE_SIZE, y)
        goal_idx = int(x) * SIDE_SIZE + int(y) * SIDE_SIZE * SIDE_SIZE
        goal_idxs.add(goal_idx)

    # possible_tiles = [position_idx for position_idx in range(token_mdp.number_of_tiles) if position_idx not in token_mdp._walls]
    # TODO: check walls; if walls: return -inf
    import ipdb; ipdb.set_trace()
    option_set = list(sorted(goal_idxs))

    xs = [10 + 10 * x for x in range(1000)]
    possible_box_positions = list(itertools.combinations([0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE, SIDE_SIZE * SIDE_SIZE - 1, ], 2))
    learner = learners.q_learning.QLearning(env=token_mdp, options=[])

    option_sets_scores = {}
    options = [options_utils.goal_to_policy(learner, goal_idx, token_mdp) for goal_idx in option_set]
    # assert len(options) == 4
    assert list(sorted(option_set)) == option_set
    fitnesses = options_utils.eval_options(SIDE_SIZE, options, possible_box_positions, xs)

    option_names = option_set
    if plot_progress:
        option_names = " ".join(str(n) for n in sorted(option_names))
        plt.plot(xs, fitnesses)
        plt.title(option_names)
        plt.show()
    print("opt-names:", " ".join(str(n) for n in sorted(option_names)), "fitness:", fitnesses[int(budget / 10)])
    return fitnesses[int(budget / 10)]


def evolve_coords(nr_options):
    POPULATION_SIZE = 6
    SIDE_SIZE = 7
    plot_progress = False
    time_budget = 600

    reward_space_size = nr_options * 2
    possible_box_positions = list(itertools.combinations([0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                                                          SIDE_SIZE * SIDE_SIZE - 1, ], 2))
    regressor = controller.meta_controller.CMAES(
        population_size=POPULATION_SIZE,
        fitness_function=fitness_simple_boxes,
        reward_space_size=reward_space_size,
        default_args=[SIDE_SIZE, nr_options, possible_box_positions, time_budget, plot_progress],
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
