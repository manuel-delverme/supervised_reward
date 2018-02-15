import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def get_weight_evolution_fitness_fn(SIDE_SIZE):
    # sweep the state space
    # create n options as sorted(mask.dot(reward_vector))[:n]
    REWARD_SPACE_SIZE = 9
    TEST_MAX_STEPS_EVAL = 1000
    possible_box_positions = list(itertools.combinations([
        0,
        SIDE_SIZE - 1,
        (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
        SIDE_SIZE * SIDE_SIZE - 1,
    ], 2))

    fake_world = e.BoxWorldSimple(side_size=SIDE_SIZE)
    fake_world.step = None
    sensor_readings = gather_sensor_readings(fake_world.number_of_tiles, fake_world._walls, fake_world.width)

    return sensor_readings, REWARD_SPACE_SIZE, TEST_MAX_STEPS_EVAL, possible_box_positions


def fitness_simple_boxes(args):
    reward_vector, SIDE_SIZE, sensor_readings, nr_options, possible_box_positions, TEST_MAX_STEPS_EVAL, plot_progress = args
    number_of_tiles = SIDE_SIZE * SIDE_SIZE
    xs = [10 + 10 * x for x in range(200)]
    options = options_utils.select_options(SIDE_SIZE, nr_options, number_of_tiles, reward_vector, sensor_readings)

    # eval options
    fitnesses = options_utils.eval_options(SIDE_SIZE, TEST_MAX_STEPS_EVAL, options, possible_box_positions, xs)
    option_names = options_utils.name_options(options)

    if plot_progress:
        option_names = " ".join(str(n) for n in sorted(option_names))
        plt.plot(xs, fitnesses)
        plt.title(option_names)
        plt.show()
    return fitnesses[-1]

def evolve_weights(nr_options):
    POPULATION_SIZE = 6
    SIDE_SIZE = 7
    plot_progress = False
    sensor_readings, reward_space_size, TEST_MAX_STEPS_EVAL, possible_box_positions = get_weight_evolution_fitness_fn(
        SIDE_SIZE)
    regressor = controller.meta_controller.CMAES(
        population_size=POPULATION_SIZE,
        fitness_function=fitness_simple_boxes,
        reward_space_size=reward_space_size,
        default_args=[
            SIDE_SIZE, sensor_readings, nr_options, possible_box_positions, TEST_MAX_STEPS_EVAL, plot_progress
        ],
    )
    return regressor.optimize()


def main():
    scores = {
        4: evolve_weights(nr_options=4)
    }
    # for nr_options in range(6):
    #    scores[nr_options] = evolve_weights(nr_options)


if __name__ == "__main__":
    main()
