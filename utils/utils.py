import learners.q_learning
import numpy as np
from utils import disk_utils


def eval_options(load_env, options, possible_box_positions, xs):
    cum_cum_reward = np.zeros(len(xs))
    for eval_step, box_positions in enumerate(possible_box_positions):
        option_set_scores = eval_option_on_mdp(load_env, box_positions, options, xs)
        cum_cum_reward += np.array(option_set_scores)
    return cum_cum_reward / (eval_step + 1)


# @disk_utils.disk_cache
def eval_option_on_mdp(load_fn, box_positions, option_vec, xs):
    mdp = load_fn(box_positions)
    learner = learners.q_learning.QLearning(env=mdp, options=option_vec)
    _, _, fitnesses = learner.learn(xs=xs)
    return fitnesses


def print_statistics(fitness, options):
    option_names = []
    for option in options:
        option_names.append(int(np.argwhere(option == -1)[0]))
    option_names = " ".join(str(n) for n in sorted(option_names))
    print("score:\t{}\toptions: {}\t{}".format(fitness, len(options), option_names))
