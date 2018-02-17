import collections
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import disk_utils
import envs.gridworld
import envs.hungry_thirsty
import envs.simple_boxes
import learners.double_q_learning
import learners.q_learning
import options_utils


@disk_utils.disk_cache
def bruteforce_options4():
    NUMBER_OF_OPTIONS = 4
    SIDE_SIZE = 7
    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=SIDE_SIZE)

    possible_tiles = [position_idx for position_idx in range(token_mdp.number_of_tiles) if position_idx not in token_mdp._walls]
    option_sets = itertools.combinations([None] * NUMBER_OF_OPTIONS + possible_tiles, NUMBER_OF_OPTIONS)
    option_sets = list(option_sets)
    random.shuffle(option_sets)

    xs = [10 + 10 * x for x in range(1000)]
    possible_box_positions = list(itertools.combinations([0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                                                          SIDE_SIZE * SIDE_SIZE - 1, ], 2))

    learner = learners.q_learning.QLearning(env=token_mdp, options=[])

    option_map = {tuple(): tuple()}
    for goal_idx in range(token_mdp.number_of_tiles):
        option_map[goal_idx] = options_utils.goal_to_policy(learner, goal_idx, token_mdp)

    option_sets_scores = {}
    for option_set in tqdm.tqdm(option_sets):
        options = [option_map[goal_idx] for goal_idx in option_set if goal_idx is not None]
        option_sets_scores[option_set] = options_utils.eval_options(SIDE_SIZE, options, possible_box_positions, xs)
    return option_sets_scores


def plot_option_scores():
    scores = bruteforce_options4()

    top_scorers = {}
    worst_scorers = {}
    score_history = {}
    import ipdb; ipdb.set_trace()
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


if __name__ == "__main__":
    plot_option_scores()
