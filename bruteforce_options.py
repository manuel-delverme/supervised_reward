import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from utils import disk_utils, options_utils
import envs.gridworld
import envs.hungry_thirsty
import envs.simple_boxes
import learners.double_q_learning
import learners.q_learning
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uniform_food_position,', action='store_true')


opt = parser_args()


@disk_utils.disk_cache
def get_no_option_score(time_budget):
    scores = bruteforce_options()
    return scores[(None, None, None, None)][int(time_budget / 10)]


@disk_utils.disk_cache
def bruteforce_options():
    NUMBER_OF_OPTIONS = 4
    SIDE_SIZE = 7
    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=SIDE_SIZE)

    possible_tiles = [position_idx for position_idx in range(token_mdp.number_of_tiles) if
                      position_idx not in token_mdp._walls]
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

    # import ipdb, ipdb.set_trace()
    option_sets_scores = {}
    for option_set in tqdm.tqdm(option_sets):
        options = [option_map[goal_idx] for goal_idx in option_set if goal_idx is not None]
        option_sets_scores[option_set] = options_utils.eval_options(SIDE_SIZE, options, possible_box_positions, xs)
    return option_sets_scores


@disk_utils.disk_cache
def bruteforce_options_complex_world():
    NUMBER_OF_OPTIONS = 4
    SIDE_SIZE = 7
    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=SIDE_SIZE, composite_actions=True)

    possible_tiles = [position_idx for position_idx in range(token_mdp.number_of_tiles) if
                      position_idx not in token_mdp._walls]
    option_sets = itertools.combinations([None] * NUMBER_OF_OPTIONS + possible_tiles, NUMBER_OF_OPTIONS)
    option_sets = list(option_sets)
    random.shuffle(option_sets)

    xs = [10 + 10 * x for x in range(500)]
    possible_box_positions = list(itertools.combinations(
        [
            0,
            SIDE_SIZE - 1,
            (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
            SIDE_SIZE * SIDE_SIZE - 1,
        ], 2))
    learner = learners.q_learning.QLearning(env=token_mdp, options=[])

    option_map = {tuple(): tuple()}
    for goal_idx in range(token_mdp.number_of_tiles):
        option_map[goal_idx] = options_utils.goal_to_policy(learner, goal_idx, token_mdp)

    option_sets_scores = {}
    for option_set in tqdm.tqdm(option_sets):
        options = [option_map[goal_idx] for goal_idx in option_set if goal_idx is not None]
        option_sets_scores[option_set] = options_utils.eval_options_on_complex_mdp(
            SIDE_SIZE, options, possible_box_positions, xs
        )
    return option_sets_scores


def plot_option_scores():
    xs_full = [10 + 10 * x for x in range(1000)]
    print("bruteforce_options")
    # scores = bruteforce_options()
    scores = bruteforce_options_complex_world()
    print("top_worst_scores")
    no_options = scores[(None, None, None, None)]
    percentiles_ranges = [1, 50, 90, 95, 99, 99.9, 100]
    fig = plt.figure(1)

    for xs in (list(range(10, 510, 10)), list(range(500, 1010, 10)), list(range(1000, 5010, 10))):
        print("get_score_history")
        score_history = get_score_history(scores, xs_full, xs)

        percentiles = [{} for nr_iter in percentiles_ranges]
        for nr_iter in tqdm.tqdm(xs, desc="percentiels"):
            z = np.percentile(score_history[nr_iter], percentiles_ranges)
            for idx, perc in enumerate(percentiles_ranges):
                percentiles[idx][nr_iter] = z[idx]

        plot_percentiles_vs_no_options(percentiles, percentiles_ranges, no_options, xs_full, xs)
        fig.clear()


def plot_distribution_vs_no_options(no_options, score_history, xs):
    X, Y = [], []
    for nr_iter in tqdm.tqdm(score_history, desc="plot distribution"):
        for s in score_history[nr_iter]:
            X.append(nr_iter)
            Y.append(s)
    print("plotting")
    domain = int(max(X) / 10) + 1
    plt.plot(range(0, 501, 10), no_options[:domain], label="no options")
    plt.scatter(X, Y, label="distr", alpha=0.05, s=0.5, antialiased=False)
    plt.legend(loc='upper left')
    plt.title("percentiles and no options")
    plt.savefig("distr_and_no_options{}.png".format(str(xs[0]) + str(xs[-1])))
    print("plotted")


def plot_p_noopt_better_opts(no_options, score_history):
    p_above = []
    for nr_iter in tqdm.tqdm(sorted(score_history.keys()), desc="plot distribution"):
        above_no_opt = 0
        below_no_opt = 0
        no_opt_score = no_options[nr_iter // 10]
        for option_set_score in score_history[nr_iter]:
            if option_set_score < no_opt_score:
                below_no_opt += 1
            else:
                above_no_opt += 1
        optset_total = (below_no_opt + above_no_opt)
        above_no_opt /= optset_total
        below_no_opt /= optset_total
        p_above.append(1 + above_no_opt)
    p_above = p_above[10:]
    plt.ylim(np.log(min(p_above)), np.log(max(p_above)))
    plt.plot(np.log(p_above))
    plt.savefig("p_above.png")
    print(p_above)


def plot_percentiles_vs_no_options(percentiles, percentiles_ranges, no_options, xs_full, xs):
    visible_no_options = []
    # xs_full = [10 + 10 * x for x in range(1000)]
    for idx, x in enumerate(xs_full):
        if x in xs:
            visible_no_options.append(no_options[idx])

    for percentile_idx, perc in tqdm.tqdm(enumerate(percentiles_ranges), desc="plot percs"):
        ys = []
        for iter_budget in xs_full:
            if iter_budget in xs:
                ys.append(percentiles[percentile_idx][iter_budget])
        # plt.plot(x_labels, ys, '-', label="perc:" + str(perc))
        plt.plot(xs, ys, label="percentile:" + str(perc), alpha=0.7)

    plt.legend(loc='upper right')
    plt.title("percentiles")
    # plt.savefig("percentiles{}.png".format("_".join(str(r) for r in percentiles_ranges)))
    plt.set_cmap("jet")
    plt.plot(xs, visible_no_options, label="no options", color="red")
    plt.legend(loc='upper left')
    plt.title("percentiles and no options")
    plt.savefig("percentiles_and_no_options{}.png".format("_".join(str(r) for r in (xs[0], xs[-1]))))
    return percentiles, percentiles_ranges


def plot_best_option_distribution(percentiles, percentiles_ranges, scores, xs, xs_full):
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
    import pickle
    with open("best_sets.pkl", "wb") as fout:
        pickle.dump(best_sets, fout)


# @disk_utils.disk_cache
def get_score_history(scores, xs_full, xs):
    xs = set(xs)
    score_history = {}
    for score_idx, nr_iter in enumerate(xs):
        score_history[nr_iter] = []
    for option_ids, option_scores in tqdm.tqdm(scores.items(), desc="score history"):
        for score_idx, nr_iter in enumerate(xs_full):
            if nr_iter in xs:
                option_score = option_scores[score_idx]
                score_history[nr_iter].append(option_score)
    return score_history


@disk_utils.disk_cache
def top_worst_scorers(scores, xs):
    top_scorers = {}
    worst_scorers = {}
    for option_ids, option_scores in scores.items():
        for score_idx, nr_iter in enumerate(xs):
            option_score = option_scores[score_idx]
            top_scorers[nr_iter] = (option_ids, option_score)
            worst_scorers[nr_iter] = (option_ids, option_score)
    for option_ids, option_scores in tqdm.tqdm(scores.items()):
        for score_idx, nr_iter in enumerate(xs):
            option_score = option_scores[score_idx]
            if top_scorers[nr_iter][1] < option_score:
                top_scorers[nr_iter] = (option_ids, option_score)
            if worst_scorers[nr_iter][1] > option_score:
                worst_scorers[nr_iter] = (option_ids, option_score)
    return top_scorers, worst_scorers


if __name__ == "__main__":
    plot_option_scores()
