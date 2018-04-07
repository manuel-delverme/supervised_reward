import itertools
import numpy as np
import matplotlib.pyplot as plt
import controller.meta_controller
import controller.meta_controller
import envs.simple_boxes
import learners
import options_utils
import functools
import bruteforce_options


@functools.lru_cache(maxsize=102)
def fitness_simple_boxes(args):
    args = dict(args)
    coords_vector = np.array(args['parameter']).reshape(-1, 2)
    SIDE_SIZE = args['SIDE_SIZE']
    budget = args['time_budget']

    assert budget % 10 == 0

    goal_idxs = set()
    for x, y in coords_vector:
        goal_idx = x + y * SIDE_SIZE
        goal_idxs.add(goal_idx)
    option_set = tuple(sorted(goal_idxs))
    fitness = eval_option_set(SIDE_SIZE, option_set, budget=budget)
    return fitness


@functools.lru_cache(maxsize=1024)
def eval_option_set(SIDE_SIZE, option_set, budget=False):
    token_mdp = envs.simple_boxes.BoxWorldSimple
    xs = [10 + 10 * x for x in range(1000)]
    if token_mdp._walls.intersection(option_set):
        fitnesses = [None for x in xs]
    else:
        possible_box_positions = list(
            itertools.combinations([0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE, SIDE_SIZE * SIDE_SIZE - 1, ],
                                   2))
        token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=SIDE_SIZE)
        learner = learners.q_learning.QLearning(env=token_mdp, options=[])
        options = tuple(options_utils.goal_to_policy(learner, goal_idx, token_mdp) for goal_idx in option_set)
        assert tuple(sorted(option_set)) == option_set
        fitnesses = options_utils.eval_options(SIDE_SIZE, options, possible_box_positions, xs)
    if budget:
        fitnesses = fitnesses[int(budget/10)]
        return fitnesses
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
        reward_space_size=reward_space_size,
    )
    import glob
    try:
        experiment_id = max([int(n[15:-4]) for n in glob.glob("fitness_history*.png")]) + 1
    except:
        experiment_id = 0
    return regressor.optimize(
        n_iterations=1000,
        fitness_function=fitness_simple_boxes,
        mdp_parameters={
            'SIDE_SIZE': SIDE_SIZE,
            'nr_options': nr_options,
            'possible_box_positions': possible_box_positions,
            'time_budget': time_budget,
            'plot_progress': plot_progress
        },
        baseline=bruteforce_options.get_no_option_score(time_budget),
        experiment_id=experiment_id,
    ),


def main():
    scores = {
        4: evolve_coords(nr_options=4),
        3: evolve_coords(nr_options=3),
        2: evolve_coords(nr_options=2),
        1: evolve_coords(nr_options=1),
    }
    print(scores)


if __name__ == "__main__":
    main()
