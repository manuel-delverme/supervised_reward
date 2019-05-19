import numpy as np

import config
import learners.approx_q_learning
from utils import disk_utils


def eval_options(env, options):
    cum_score = 0
    for eval_step in range(config.repeat_eval_options):
        cum_score += eval_option_on_mdp(env, options)
    return cum_score / (eval_step + 1)


# @disk_utils.disk_cache
def eval_option_on_mdp(env, options):
    _, _, fitness, _ = learners.approx_q_learning.learn(environment=env, options=options, training_steps=config.option_eval_training_steps, eval_fitness=True)
    return fitness


def print_statistics(fitness, options):
    option_names = []
    for option in options:
        option_names.append(int(np.argwhere(option == -1)[0]))
    option_names = " ".join(str(n) for n in sorted(option_names))
    print("score:\t{}\toptions: {}\t{}".format(fitness, len(options), option_names))


def to_tuple(img):
    b = tuple(tuple(tuple(column) for column in row) for row in img)
    return b