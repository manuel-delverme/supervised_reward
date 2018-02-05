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


def evolve_weights():
    POPULATION_SIZE = 6  # if < 6 cmaes mirrors, disable that (?)
    SIDE_SIZE = 6

    fitness_fn, reward_space_size = envs.simple_boxes.BoxWorldSimple.get_weight_evolution_fitness_fn(SIDE_SIZE), 4

    regressor = controller.meta_controller.CMAES(
        population_size=POPULATION_SIZE,
        fitness_function=fitness_fn,
        reward_space_size=reward_space_size,
    )
    regressor.optimize()


def evolve_intrinsic():
    POPULATION_SIZE = 6  # if < 6 cmaes mirrors, disable that (?)
    TRAINING_NO_CHANGE_STOP = 1000
    GENERATE_RANDOM_OPTIONS = False
    TRAINING_MAX_STEPS = 10000

    TEST_MAX_STEPS_TRAIN = 2000
    TEST_MAX_STEPS_EVAL = 1000
    OPTION_LEARNING_STEPS = 10000
    SIDE_SIZE = 6

    # fitness_fn, reward_space_size = envs.hungry_thirsty.get_env_fitness_fn(SIDE_SIZE)
    # fitness_fn, reward_space_size = envs.boxes.BoxWorld.get_fitness_fn(SIDE_SIZE), 18

    fitness_fn, reward_space_size = envs.simple_boxes.BoxWorldSimple.get_fitness_fn(SIDE_SIZE), 9

    regressor = controller.meta_controller.CMAES(
        population_size=POPULATION_SIZE,
        fitness_function=fitness_fn,
        reward_space_size=reward_space_size,
    )
    regressor.optimize()


@disk_utils.disk_cache
def bruteforce_options():
    scores = {}
    number_of_options = 4
    TEST_MAX_STEPS_EVAL = 500
    SIDE_SIZE = 6
    scores = collections.defaultdict(dict)

    option_sets = itertools.combinations([None] * (number_of_options) + list(range(36)), number_of_options)
    option_sets = list(option_sets)
    random.shuffle(option_sets)
    possible_box_positions = list(itertools.combinations([0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                                                          SIDE_SIZE * SIDE_SIZE - 1, ], 2))

    xs = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 10000]
    dxs = [xs[0], ] + [x - xs[idx] for idx, x in enumerate(xs[1:])]

    progress = tqdm.tqdm(total=len(option_sets) * len(xs))

    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=(1, 2))
    learner = learners.double_q_learning.QLearning(env=token_mdp, options=[], test_run=True)

    option_map = {tuple(): tuple()}

    for o in range(36):
        token_mdp.agent_position_idx = o
        learner.generate_option()
        option_vec = tuple(learner.available_actions[-1])
        option_map[o] = option_vec
        # token_mdp.show_board(policy=np.array(option_vec), highlight_square=o)

    option_sets = [tuple(o) for o in option_sets]
    option_sets = [tuple(o for o in option_set if o is not None) for option_set in option_sets]
    option_vecs = [tuple(option_map[o] for o in option_set) for option_set in option_sets]

    # import ipdb; ipdb.set_trace()
    for option_ids, option_vec in zip(option_sets, option_vecs):
        cum_scores = collections.defaultdict(float)
        for eval_step, box_positions in enumerate(possible_box_positions):
            option_set_score = eval_option_on_mdp(TEST_MAX_STEPS_EVAL, box_positions, option_vec, dxs)

            for k in option_set_score.keys():
                cum_scores[k] += option_set_score[k]
            progress.update(1)
        scores[option_ids] = dict(cum_scores)
        # print_statistics(fitness, option_set)
    return scores


# @disk_utils.disk_cache
# @numba.jit
def eval_option_on_mdp(TEST_MAX_STEPS_EVAL, box_positions, option_vec, dxs):
    option_set_score = {}
    mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=box_positions)
    learner = learners.q_learning.QLearning(env=mdp, options=option_vec, test_run=True)

    # mdp.show_board(some_matrix=learner.qmax, policy=learner.qargmax, info={"desc":"cached q"})
    # mdp.show_board(some_matrix=learner.Q, policy=learner.qargmax, info={"desc":"real q"})

    training_steps = 0
    for test_max_steps_train in dxs:
        learner.learn(max_steps=test_max_steps_train)

        training_steps += test_max_steps_train

        cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL, render=False)
        # if cum_reward > 0:
        #     cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL, render=True)
        option_set_score[test_max_steps_train] = cum_reward
        # if cum_reward > 0:
        #     mdp.show_board(some_matrix=learner.qmax, policy=learner.qargmax, option_vec=option_vec,
        #  info={"training_steps", training_steps})
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
    evolve_weights()
    # evolve_intrinsic()

    # update the genetic search, and plot where the masks activate

    # test_qlearning()
    # plot_option_scores()
