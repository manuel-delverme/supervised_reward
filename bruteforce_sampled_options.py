import collections
import math
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
import os


def hack():
    plt.gcf().clear()


plt.clear = hack


@disk_utils.disk_cache
def bruteforce_option_stats():
    NUMBER_OF_OPTIONS = 4
    SIDE_SIZE = 7
    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=SIDE_SIZE)

    possible_tiles = [position_idx for position_idx in range(token_mdp.number_of_tiles) if position_idx not in token_mdp._walls]
    option_sets = itertools.combinations([None] * NUMBER_OF_OPTIONS + possible_tiles, NUMBER_OF_OPTIONS)
    option_sets = list(option_sets)
    random.shuffle(option_sets)
    possible_box_positions = list(itertools.combinations([0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE, SIDE_SIZE * SIDE_SIZE - 1, ], 2))
    learner = learners.q_learning.QLearning(env=token_mdp, options=[])
    xs = [50, 60, 100]
    nr_samples = 100

    option_map = {tuple(): tuple()}
    for goal_idx in range(token_mdp.number_of_tiles):
        option_map[goal_idx] = options_utils.goal_to_policy(learner, goal_idx, token_mdp)

    option_sets_scores = {}
    for option_set in tqdm.tqdm(option_sets):
        options = [option_map[goal_idx] for goal_idx in option_set if goal_idx is not None]
        option_sets_scores[option_set] = options_utils.gather_option_stats(SIDE_SIZE, options, possible_box_positions, xs, nr_samples)

    return option_sets_scores


def plot_option_scores():
    xs = [10 + 10 * x for x in range(1000)]

    scores = bruteforce_option_stats()
    top_scorers = {}
    worst_scorers = {}
    score_history = {}
    # for option_ids, option_scores in list(scores.items()):
    # 	scores[option_ids] = np.log(1 + option_scores)
    for option_ids, option_scores in scores.items():
        for score_idx, nr_iter in enumerate(xs):
            option_score = option_scores[score_idx]
            top_scorers[nr_iter] = (option_ids, option_score)
            worst_scorers[nr_iter] = (option_ids, option_score)
            score_history[nr_iter] = []
        break
    for option_ids, option_scores in tqdm.tqdm(scores.items()):
        for score_idx, nr_iter in enumerate(xs):
            option_score = option_scores[score_idx]
            if top_scorers[nr_iter][1] < option_score:
                top_scorers[nr_iter] = (option_ids, option_score)
            if worst_scorers[nr_iter][1] > option_score:
                worst_scorers[nr_iter] = (option_ids, option_score)
            score_history[nr_iter].append(option_score)
    scores = bruteforce_options()
    no_options = scores[(None, None, None, None)]

    xs = [10, 30, 60, 200, 500, 1000, 2000, 4000, 8000, xs[-1]]
    import seaborn as sns
    sns.set(color_codes=True)
    data = []
    # xs = sorted(score_history.keys())
    for x in xs:
        data.append(score_history[x])

    plt.figure(1, figsize=(15, 5))
    y_low, y_high = worst_scorers[xs[0]][1], top_scorers[xs[-1]][1]
    y_high = np.log(1000)
    plt.ylim(ymin=y_low, ymax=y_high)
    # plt.subplot(
    data = np.array(data)

    no_options_show = []
    xs_full = [10 + 10 * x for x in range(1000)]
    for idx, x in enumerate(xs_full):
        if x in xs:
            no_options_show.append(no_options[idx])

    percentiles_ranges = [1, 50, 90, 95, 99, 99.9, 100]
    percentiles = [{} for nr_iter in percentiles_ranges]
    for nr_iter in tqdm.tqdm(xs, desc="percentiels"):
        for idx, perc in enumerate(percentiles_ranges):
            percentiles[idx][nr_iter] = np.percentile(score_history[nr_iter], perc)

    x_labels = [str(idx) + "_" + str(x) + "_iter" for idx, x in enumerate(xs)]
    # plt.plot(x_labels, data)
    plt.plot(x_labels, data.mean(axis=1), 'o', label="mean")

    # print(percentiles)
    for idx, perc in tqdm.tqdm(enumerate(percentiles_ranges), desc="plot percs"):
        ys = [percentiles[idx][x] for x in xs]
        plt.plot(x_labels, ys, 'o', label="perc:" + str(perc))
    plt.legend(loc='upper right')
    plt.title("percentiles")
    plt.savefig("percentiles.png")

    plt.plot(x_labels, no_options_show, label="no options")
    plt.legend(loc='upper left')
    plt.title("percentiles and no options")
    plt.savefig("percentiles_and_no_options.png")

    plt.clear()

    x_labels = [str(idx) + "_" + str(x) + "%" for idx, x in enumerate(percentiles_ranges)]
    for nr_iter in tqdm.tqdm(percentiles[0].keys(), desc="perc vs score"):
        ys = [percentiles[percentiles_ranges.index(perc)][nr_iter] for perc in percentiles_ranges]
        plt.plot(x_labels, ys, 'o', label="iter:" + str(nr_iter))
    plt.legend(loc='upper left')
    plt.title("perc vs score")
    plt.savefig("perc_vs_score.png")
    plt.clear()

    x_labels = [str(idx) + "_" + str(x) + "_iter" for idx, x in enumerate(xs)]

    ys = [percentiles[percentiles_ranges.index(100)][nr_iter] for nr_iter in xs]
    plt.plot(x_labels, ys, '-', label="best")
    plt.legend(loc='upper left')
    plt.title("best")
    plt.savefig("best.png")
    # plt.clear()

    plt.plot(x_labels, no_options_show, label="no options")
    plt.legend(loc='upper left')
    plt.title("best and no options")
    plt.savefig("best_and_no_options.png")
    plt.clear()

    # ys = [percentiles[percentiles_ranges.index(1)][nr_iter] for nr_iter in xs]
    # plt.plot(x_labels, ys, label="perc:1%")

    # ys = [percentiles[percentiles_ranges.index(90)][nr_iter] for nr_iter in xs]
    # plt.plot(x_labels, ys, label="perc:90%")

    ys = [percentiles[percentiles_ranges.index(50)][nr_iter] for nr_iter in xs]
    plt.plot(x_labels, ys, label="perc:50%")
    plt.legend(loc='upper left')
    plt.title("perc:50%")
    plt.savefig("perc:50%.png")
    plt.clear()

    # ys = [percentiles[percentiles_ranges.index(95)][nr_iter] for nr_iter in xs]
    # plt.plot(x_labels, ys, label="perc:95%")

    ys = [percentiles[percentiles_ranges.index(99)][nr_iter] for nr_iter in xs]
    plt.plot(x_labels, ys, label="perc:99%")
    plt.legend(loc='upper left')
    plt.title("perc:99%")
    plt.savefig("perc:99%.png")
    plt.clear()

    # plt.legend(loc='upper left')
    # plt.show()

    cutoff = {nr_iter: percentiles[percentiles_ranges.index(99.9)][nr_iter] for nr_iter in xs}
    best_sets = {}

    for nr_iter in xs:
        best_sets[nr_iter] = []

    print("best sets")
    for option_ids, option_scores in tqdm.tqdm(scores.items(), desc="best sets"):
        for iter_idx, option_score in enumerate(option_scores):
            nr_iter = xs_full[iter_idx]
            if nr_iter in xs:
                if option_score > cutoff[nr_iter]:
                    best_sets[nr_iter].append((option_ids, option_score))

    for nr_iter in best_sets.copy().keys():
        best_sets[nr_iter].sort(key=lambda _x: -_x[1])

    import pprint
    pprint.pprint(best_sets)
    import pickle
    with open("best_sets.txt", "wb") as fout:
        pickle.dump(pprint.pformat(best_sets), fout)

    # mdp = envs.boxes.BoxWorld(side_size=6, box_positions=())
    # mdp.show_board(just_numbers=True)
    # input("enjoy")

    # df = pd.DataFrame(data, index=xs)
    # df.groupby(axis=1)

    # df.describe()
    # ax = sns.tsplot(data=data, ci=[50, 90], color="m")
    # return df


if __name__ == "__main__":
    plot_option_scores()
