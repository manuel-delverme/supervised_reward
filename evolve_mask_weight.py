import controller.meta_controller
import matplotlib.pyplot as plt
import itertools
import numpy as np
import controller.meta_controller
import envs.gridworld
import envs.hungry_thirsty
import envs.simple_boxes as e
import learners.double_q_learning
import learners.q_learning


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


def get_weight_evolution_fitness_fn(SIDE_SIZE, nr_options, plot_progress=False):
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

    def fitness_simple_boxes(reward_vector):
        scores = []

        for goal_idx, position_idx in enumerate(range(fake_world.number_of_tiles)):
            sensor_reading = sensor_readings[position_idx]
            score = np.sum(reward_vector[sensor_reading])
            scores.append((goal_idx, score))

        xs = [10 + 10 * x for x in range(200)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        options = []

        for goal_idx, goal_score in scores[:nr_options]:
            option = tuple(learners.q_learning.learn_option(goal_idx, e.BoxWorldSimple(side_size=SIDE_SIZE)))
            options.append(option)
            options = sorted(options, key=lambda x: x.index(-1))

        # eval options
        cum_cum_reward = np.zeros(len(xs))
        for eval_step, box_positions in enumerate(possible_box_positions):
            option_set_scores = e.BoxWorldSimple.eval_option_on_mdp(SIDE_SIZE, TEST_MAX_STEPS_EVAL, box_positions,
                                                                    options, xs)
            # print(option_set_scores)
            cum_cum_reward += np.array(option_set_scores)

        if plot_progress:
            #  print(reward_vector, end="")
            option_names = []
            for option in options:
                for idx, action in enumerate(option):
                    if action == -1:
                        option_names.append(idx)
                        break

            # BoxWorldSimple.HACK += 1
            # if BoxWorldSimple.HACK % 10 == 0:
            #     fake_world.box_positions = (-1, -1)
            #     fake_world.agent_position_idx = -1
            #     opt_ids = [s[0] for s in scores][1:fitnesses.index(max(fitnesses))]
            #     fake_world.show_board(highlight_squares=opt_ids, info={'score': max(fitnesses)})
            option_names = " ".join(str(n) for n in sorted(option_names))
            ys = cum_cum_reward / (eval_step + 1)
            plt.plot(xs, ys)
            plt.title(option_names)
            plt.show()
        return cum_cum_reward[-1] / (eval_step + 1)

    return fitness_simple_boxes, REWARD_SPACE_SIZE


def evolve_weights(nr_options):
    POPULATION_SIZE = 6
    SIDE_SIZE = 7
    fitness_fn, reward_space_size = get_weight_evolution_fitness_fn(SIDE_SIZE, nr_options)

    regressor = controller.meta_controller.CMAES(
        population_size=POPULATION_SIZE,
        fitness_function=fitness_fn,
        reward_space_size=reward_space_size,
    )
    return regressor.optimize()


if __name__ == "__main__":
    scores = {}
    # for nr_options in range(6):
    #    scores[nr_options] = evolve_weights(nr_options)
    scores[4] = evolve_weights(nr_options=5)
