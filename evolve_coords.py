import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import learners
import controller.meta_controller
import itertools
import numpy as np
import controller.meta_controller
import envs.simple_boxes as e
import options_utils


def gather_sensor_readings(world_tiles, world_walls, world_width):
    sensor_readings = [None, ] * world_tiles

    for position_idx in range(world_tiles):
        if position_idx in world_walls:
            continue
        # tmp_world.show_board(highlight_square=position_idx)
        can_reach = np.ones(shape=(3, 3), dtype=np.bool) * False

        can_reach[1][1] = True

        x0 = position_idx % world_width
        y0 = position_idx // world_width
        for y_shift in (-1, 0, 1):
            for x_shift in (-1, 0, 1):
                x = x0 + x_shift
                y = y0 + y_shift
                pos_shift = position_idx + x_shift + y_shift * world_width
                # print("\n{},{} ({}) -> {},{} ({})".format(x0, y0, position_idx, x, y, pos_shift), end='')
                if x < 0 or x >= world_width:
                    continue
                if y < 0 or y >= world_width:
                    continue
                if pos_shift in world_walls:
                    continue
                # print(" can_reach", end='')
                can_reach[y_shift + 1, x_shift + 1] = True

        sensor_readings[position_idx] = can_reach.flatten()
    return sensor_readings


def fitness_simple_boxes(args):
    coords_vector, SIDE_SIZE, nr_options, possible_box_positions, budget, plot_progress = args
    assert budget % 10 == 0
    xs = [10 + 10 * x for x in range(10000)]

    options = []
    for x, y in list(zip(coords_vector, coords_vector[1:]))[::2]:
        x = max(0, x)
        y = max(0, y)
        goal_idx = int(x) * SIDE_SIZE + int(y) * SIDE_SIZE * SIDE_SIZE
        option = tuple(learners.q_learning.learn_option(goal_idx, e.BoxWorldSimple(side_size=SIDE_SIZE)))
        options.append(option)

    # eval options
    fitnesses = options_utils.eval_options(SIDE_SIZE, options, possible_box_positions, xs)
    option_names = options_utils.name_options(options)

    if plot_progress:
        option_names = " ".join(str(n) for n in sorted(option_names))
        plt.plot(xs, fitnesses)
        plt.title(option_names)
        plt.show()
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
