import collections
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import controller.meta_controller
import disk_utils
import envs.gridworld
import envs.hungry_thirsty
import envs.simple_boxes
import learners.double_q_learning
import learners.q_learning


def main():
    POPULATION_SIZE = 4
    TRAINING_NO_CHANGE_STOP = 1000
    GENERATE_RANDOM_OPTIONS = False
    TRAINING_MAX_STEPS = 10000

    TEST_MAX_STEPS_TRAIN = 2000
    TEST_MAX_STEPS_EVAL = 1000
    SIDE_SIZE = 6

    env_name = "simple_boxworld"  # "hungry-thirsty"

    if env_name == "hungry-thirsty":
        def fitness_hungry_thirsty(reward_vector):
            # init a world
            possible_box_positions = [
                0,
                SIDE_SIZE - 1,
                SIDE_SIZE * SIDE_SIZE - 1,
                (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
            ]
            _box_positions = []
            for idx, box_pos in enumerate(possible_box_positions[:-1]):
                _box_positions.append((box_pos, possible_box_positions[idx + 1]))
                _box_positions.append((possible_box_positions[idx + 1], box_pos))

            random.shuffle(_box_positions)
            possible_box_positions = (p for p in _box_positions)

            water_pos, food_pos = next(possible_box_positions)

            if GENERATE_RANDOM_OPTIONS:
                options = pick_random_options()
            else:
                print("training with water: {} food {}".format(water_pos, food_pos))
                mdp = envs.hungry_thirsty.HungryThirsty(
                    side_size=SIDE_SIZE, water_position=water_pos, food_position=food_pos
                )

                # define an intrinsic reward fn
                def intrinsic_reward_function(_mdp):
                    thirst = _mdp._state['thirsty']
                    hunger = _mdp._state['hungry']
                    x = np.array((
                        thirst and hunger,
                        not thirst and not hunger,
                        thirst and not hunger,
                        hunger and not thirst,
                    ), dtype=np.int)
                    # TODO: should be optimized as reward_vec[idx]
                    return np.dot(reward_vector, x)

                # generate option set
                learner = learners.double_q_learning.QLearning(
                    env=mdp,
                    surrogate_reward=intrinsic_reward_function,
                    train_run=True,
                )
                options, cum_reward = learner.learn(
                    steps_of_no_change=TRAINING_NO_CHANGE_STOP,
                    max_steps=TRAINING_MAX_STEPS,
                    generate_options=True
                )

            # eval options

            # cum_cum_reward += cum_reward
            # num_of_test_samples += 1

            cum_cum_reward = 0
            print_statistics(-1, options)
            for eval_step, box_positions in enumerate(possible_box_positions):
                food_pos, water_pos = box_positions

                mdp = envs.hungry_thirsty.HungryThirsty(side_size=6, food_position=food_pos, water_position=water_pos)
                learner = learners.double_q_learning.QLearning(env=mdp, options=options, test_run=True)
                _, _ = learner.learn(max_steps=TEST_MAX_STEPS_TRAIN)
                cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
                cum_cum_reward += cum_reward

            fitness = cum_cum_reward / (eval_step + 1)
            print_statistics(fitness, options)
            return fitness,

        fitness_fn = fitness_hungry_thirsty
        reward_space_size = 4
    elif env_name == "boxworld":
        def fitness_boxes(reward_vector):
            # init a world
            possible_box_positions = list(itertools.combinations([
                0,
                SIDE_SIZE - 1,
                (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                SIDE_SIZE * SIDE_SIZE - 1,
            ], 2))
            random.shuffle(possible_box_positions)
            possible_box_positions = (p for p in possible_box_positions)

            training_sample = next(possible_box_positions)

            if GENERATE_RANDOM_OPTIONS:
                options = pick_random_options()
            else:
                mdp = envs.boxes.BoxWorld(side_size=6, box_positions=training_sample)

                # define reward fn
                def intrinsic_reward_function(_mdp):
                    # thirst = _mdp._state['thirsty']
                    hunger = _mdp._state['hungry']

                    box1_pos, box2_pos = _mdp.box_positions
                    box1 = _mdp._state['box'][box1_pos]
                    box2 = _mdp._state['box'][box2_pos]
                    # world_states = []
                    _hack_idx = 0
                    for _box1 in envs.boxes._BoxState:
                        for _box2 in envs.boxes._BoxState:
                            for _hunger in (True, False):
                                # world_states.append((box1 == _box1 and box2 == _box2 and hunger == _hunger))
                                if box1 == _box1 and box2 == _box2 and hunger == _hunger:
                                    _idx = _hack_idx
                                _hack_idx += 1

                    # x = np.array(world_states, dtype=np.int)
                    # return np.dot(reward_vector, x)
                    return reward_vector[_idx]

                # generate options set
                learner = learners.double_q_learning.QLearning(env=mdp, surrogate_reward=intrinsic_reward_function,
                                                               train_run=True)
                options, cum_reward = learner.learn(steps_of_no_change=1000, max_steps=10000, generate_options=True)

            # eval options
            cum_cum_reward = 0
            for eval_step, box_positions in enumerate(possible_box_positions):
                mdp = envs.boxes.BoxWorld(side_size=6, box_positions=box_positions)
                learner = learners.double_q_learning.QLearning(env=mdp, options=options, test_run=True)
                _, _ = learner.learn(max_steps=TEST_MAX_STEPS_TRAIN, generate_options=False, plot_progress=False)

                cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
                cum_cum_reward += cum_reward
            fitness = cum_cum_reward / eval_step

            print_statistics(fitness, options)
            return fitness

        fitness_fn = fitness_boxes
        reward_space_size = 18
    elif env_name == "simple_boxworld":
        def fitness_simple_boxes(reward_vector):

            def gen_worlds():
                possible_box_positions = list(itertools.combinations([
                    0,
                    SIDE_SIZE - 1,
                    (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                    SIDE_SIZE * SIDE_SIZE - 1,
                    ], 2))
                random.shuffle(possible_box_positions)
                for p in possible_box_positions:
                    yield envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=p)
            possible_worlds = gen_worlds()

            training_world = next(possible_worlds)
            sensor_readings = gather_sensor_readings(training_world)

            # define reward fn
            def intrinsic_reward_function(_mdp):
                sensor_reading = sensor_readings[_mdp.agent_position_idx]
                assert np.allclose(np.sum(reward_vector[sensor_reading]), reward_vector.dot(sensor_reading), 1e-10)
                # return reward_vector.dot(sensor_reading)
                return np.sum(reward_vector[sensor_reading])

            # generate options set
            learner = learners.double_q_learning.QLearning(env=training_world, surrogate_reward=intrinsic_reward_function,
                                                           train_run=True)
            options, cum_reward = learner.learn(steps_of_no_change=1000, max_steps=10000, generate_options=True)

            # eval options
            cum_cum_reward = 0
            for eval_step, testing_world in enumerate(possible_worlds):
                learner = learners.double_q_learning.QLearning(env=testing_world, options=options, test_run=True)
                _, _ = learner.learn(max_steps=TEST_MAX_STEPS_TRAIN, generate_options=False)

                cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
                cum_cum_reward += cum_reward
            fitness = cum_cum_reward / eval_step

            print(reward_vector, end="")
            print_statistics(fitness, options)
            return fitness

        def gather_sensor_readings(training_world):
            sensor_readings = {}
            for position_idx in range(training_world.number_of_tiles):
                reading = np.ones(shape=(3, 3), dtype=np.bool)
                for dx in (-1, 0, +1):
                    x = position_idx % training_world.width
                    # for dy in (-mdp.width, 0, mdp.width):
                    for dy in (-1, 0, +1):
                        y = position_idx // training_world.width
                        px = dx + x
                        py = dy + y
                        if px < 0 or py < 0 or py == training_world.height or px == training_world.width:
                            # print("skipped", x, y, px, py)
                            continue
                        tile_idx = position_idx + dx + dy * training_world.width
                        # sensor_readings[position_idx].append((position_idx + dx, position_idx + dy * mdp.width))
                        if position_idx not in training_world._walls:
                            reading[dy + 1][dx + 1] = False
                        else:
                            if tile_idx not in training_world._walls[position_idx]:
                                reading[dy + 1][dx + 1] = False
                sensor_readings[position_idx] = reading.flatten()
            return sensor_readings

        fitness_fn = fitness_simple_boxes
        reward_space_size = 9
    else:
        raise NotImplementedError("{} is not a valid environment".format(env_name))

    regressor = controller.meta_controller.CMAES(
        population_size=POPULATION_SIZE,
        fitness_function=fitness_fn,
        reward_space_size=reward_space_size,
    )
    regressor.optimize()


@disk_utils.disk_cache
def bruteforce_options():
    number_of_options = 4
    TEST_MAX_STEPS_EVAL = 100
    SIDE_SIZE = 6
    scores = collections.defaultdict(dict)

    option_sets = itertools.combinations([None] * (number_of_options) + list(range(36)), number_of_options)
    option_sets = list(option_sets)
    random.shuffle(option_sets)
    possible_box_positions = list(itertools.combinations([0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                                                          SIDE_SIZE * SIDE_SIZE - 1, ], 2))

    xs = [10, 100, 500, 1000, 10000]
    dxs = [x - xs[idx] for idx, x in enumerate(xs[1:])]

    progress = tqdm.tqdm(total=len(option_sets) * len(xs))

    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=(1, 2))
    learner = learners.double_q_learning.QLearning(env=token_mdp, options=[], test_run=True)

    option_map = {
        tuple(): tuple()
    }

    for o in range(36):
        token_mdp.agent_position_idx = o
        learner.generate_option()
        option_vec = tuple(learner.available_actions[-1])
        option_map[o] = option_vec

    option_sets = [tuple(o) for o in option_sets]
    option_sets = [tuple(o for o in option_set if o is not None) for option_set in option_sets]
    option_vecs = [tuple(option_map[o] for o in option_set) for option_set in option_sets]

    for option_ids, option_vec in zip(option_sets, option_vecs):
        cum_scores = collections.defaultdict(float)
        for eval_step, box_positions in enumerate(possible_box_positions):
            option_set_score = eval_option_on_mdp(TEST_MAX_STEPS_EVAL, box_positions, option_vec, dxs)
            # fitness = cum_cum_reward / eval_step
            for k in option_set_score.keys():
                cum_scores[k] += option_set_score[k]
            progress.update(1)
        scores[option_ids] = dict(cum_scores)
        # print_statistics(fitness, option_set)
    return scores


@disk_utils.disk_cache
# @numba.jit
def eval_option_on_mdp(TEST_MAX_STEPS_EVAL, box_positions, option_vec, dxs):
    option_set_score = {}
    mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=box_positions)
    learner = learners.q_learning.QLearning(env=mdp, options=option_vec, test_run=True)

    for test_max_steps_train in dxs:
        learner.learn(max_steps=test_max_steps_train)
        cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
        option_set_score[test_max_steps_train] = cum_reward
        # cum_cum_reward += cum_reward
    return option_set_score


def print_statistics(fitness, options):
    option_names = []
    for option in options:
        for idx, action in enumerate(option):
            if action == -1:
                option_names.append(idx)
                break
    option_names = " ".join(str(n) for n in sorted(option_names))
    print("score:\t{}\toptions: {}\t{}".format(fitness, len(options), option_names))


def pick_random_options():
    mdp = envs.boxes.BoxWorld(side_size=6, box_positions=())
    options = []
    # for goal in random.sample(range(mdp.number_of_tiles), random.randrange(1, 4)):
    for goal in (0, 5, 30, 35):
        opt = learners.double_q_learning.learn_option(goal, mdp)
        # TODO: REMOVE HACK
        if opt.shape[0] < mdp.observation_space.n:
            # TODO: remove print("OPTION SIZE MISMATCH, TILING")
            opt = np.tile(
                opt[:mdp.number_of_tiles],
                mdp.observation_space.n // mdp.number_of_tiles
            )
        options.append(opt)
    return options


def plot_option_scores():
    scores = bruteforce_options()

    top_scorers = {}
    worst_scorers = {}
    score_history = {}
    for option_ids, option_scores in scores.items():
        for nr_iter, option_score in option_scores.items():
            top_scorers[nr_iter] = (option_ids, option_score)
            worst_scorers[nr_iter] = (option_ids, option_score)
            score_history[nr_iter] = []
        break
    for option_ids, option_scores in scores.items():
        for nr_iter, option_score in option_scores.items():
            if top_scorers[nr_iter][1] < option_score:
                top_scorers[nr_iter] = (option_ids, option_score)
            if worst_scorers[nr_iter][1] > option_score:
                worst_scorers[nr_iter] = (option_ids, option_score)
            score_history[nr_iter].append(option_score)

    import seaborn as sns
    sns.set(color_codes=True)
    data = []
    xs = sorted(score_history.keys())
    for x in xs:
        data.append(score_history[x])
    sum = 0
    indices = []
    for dx in xs:
        sum += dx
        indices.append(sum)

    plt.figure(1)
    y_low, y_high = worst_scorers[100][1], top_scorers[100][1]
    y_high = 1000
    plt.ylim(ymin=y_low, ymax=y_high)
    # plt.subplot(
    data = np.array(data)

    percentiles_ranges = [1, 50, 90, 95, 99, 99.9, 100]
    percentiles = [{} for nr_iter in percentiles_ranges]
    for nr_iter in xs:
        for idx, perc in enumerate(percentiles_ranges):
            percentiles[idx][nr_iter] = np.percentile(score_history[nr_iter], perc)

    x_labels = [str(idx) + "_" + str(x) + "_iter" for idx, x in enumerate(xs)]
    # plt.plot(x_labels, data)
    plt.plot(x_labels, data.mean(axis=1), 'o', label="mean")

    # print(percentiles)
    for idx, perc in enumerate(percentiles_ranges):
        ys = [percentiles[idx][x] for x in xs]
        plt.plot(x_labels, ys, 'o', label="perc:" + str(perc))
    plt.legend(loc='upper right')
    plt.show()

    plt.title("perc vs score")
    x_labels = [str(idx) + "_" + str(x) + "%" for idx, x in enumerate(percentiles_ranges)]
    for nr_iter in percentiles[0].keys():
        ys = [percentiles[percentiles_ranges.index(perc)][nr_iter] for perc in percentiles_ranges]
        plt.plot(x_labels, ys, 'o', label="iter:" + str(nr_iter))
    plt.legend(loc='upper left')
    plt.show()

    x_labels = [str(idx) + "_" + str(x) + "_iter" for idx, x in enumerate(xs)]

    ys = [percentiles[percentiles_ranges.index(100)][nr_iter] for nr_iter in xs]
    plt.plot(x_labels, ys, 'o', label="best")

    no_options = scores[()]
    ys = [no_options[x] for x in xs]
    plt.plot(x_labels, ys, label="no options")

    # ys = [percentiles[percentiles_ranges.index(1)][nr_iter] for nr_iter in xs]
    # plt.plot(x_labels, ys, label="perc:1%")

    # ys = [percentiles[percentiles_ranges.index(90)][nr_iter] for nr_iter in xs]
    # plt.plot(x_labels, ys, label="perc:90%")

    ys = [percentiles[percentiles_ranges.index(50)][nr_iter] for nr_iter in xs]
    plt.plot(x_labels, ys, label="perc:50%")

    # ys = [percentiles[percentiles_ranges.index(95)][nr_iter] for nr_iter in xs]
    # plt.plot(x_labels, ys, label="perc:95%")

    ys = [percentiles[percentiles_ranges.index(99)][nr_iter] for nr_iter in xs]
    plt.plot(x_labels, ys, label="perc:99%")

    plt.legend(loc='upper left')
    plt.show()

    cutoff = {nr_iter: percentiles[percentiles_ranges.index(99.9)][nr_iter] for nr_iter in xs}
    best_sets = {}

    for nr_iter in xs:
        best_sets[nr_iter] = []

    for option_ids, option_scores in scores.items():
        for nr_iter, option_score in option_scores.items():
            if option_score > cutoff[nr_iter]:
                best_sets[nr_iter].append((option_ids, option_score))

    for nr_iter in best_sets.copy().keys():
        best_sets[nr_iter].sort(key=lambda _x: -_x[1])

    import pprint
    pprint.pprint(best_sets)

    mdp = envs.boxes.BoxWorld(side_size=6, box_positions=())
    mdp.show_board(just_numbers=True)
    input("enjoy")


    # df = pd.DataFrame(data, index=indices)
    # df.groupby(axis=1)

    # df.describe()
    # ax = sns.tsplot(data=data, ci=[50, 90], color="m")
    # return df

def test_qlearning():
    SIDE_SIZE = 6
    TEST_MAX_STEPS_TRAIN = 2000
    TEST_MAX_STEPS_EVAL = 1000

    possible_box_positions = list(itertools.combinations(
        [0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE, SIDE_SIZE * SIDE_SIZE - 1, ], 2))
    cum_cum_reward = 0

    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=(1, 2))
    learner = learners.double_q_learning.QLearning(env=token_mdp, options=[], test_run=True)
    token_mdp.agent_position_idx = 0
    learner.generate_option()
    option_vec0 = tuple(learner.available_actions[-1])
    token_mdp.agent_position_idx = 17
    learner.generate_option()
    option_vec1 = tuple(learner.available_actions[-1])
    option_vec = [option_vec0, option_vec1]

    for eval_step, box_positions in enumerate(possible_box_positions):
        mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=box_positions)
        learner = learners.q_learning.QLearning(env=mdp, options=option_vec, test_run=True)
        learner.learn(max_steps=TEST_MAX_STEPS_TRAIN)

        cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
        cum_cum_reward += cum_reward
    fitness_q = cum_cum_reward / eval_step
    print(fitness_q)

    cum_cum_reward = 0
    for eval_step, box_positions in enumerate(possible_box_positions):
        mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=box_positions)
        learner = learners.double_q_learning.QLearning(env=mdp, options=option_vec, test_run=True)
        learner.learn(max_steps=TEST_MAX_STEPS_TRAIN, generate_options=False)

        cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
        cum_cum_reward += cum_reward
    fitness_2q = cum_cum_reward / eval_step
    print("q, 2q", fitness_q, fitness_2q)


if __name__ == "__main__":
    main()
    # test_qlearning()
    # plot_option_scores()
