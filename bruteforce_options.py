import itertools
import random
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
    parser.add_argument('--uniform_food_position', action='store_true')
    parser.add_argument('--batch_idx', type=int)
    args = parser.parse_args()
    return args


opt = parser_args()
print(opt)


@disk_utils.disk_cache
def bruteforce_options():
    # used to load old version cached file
    pass


@disk_utils.disk_cache
def bruteforce_options_complex_world(nr_of_options=4, side_size=7, complex_actions=True, use_compelx_options=False):
    # steps_to_record = [10 + 10 * x for x in range(1000)]

    steps_to_record = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 750,
                       1000]  # , 1500, 2000, 2500, 3000, 4000, 5000]
    nr_batches = 16

    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=side_size, composite_actions=True)

    possible_tiles = token_mdp.get_walkable_tiles()
    option_sets = list(itertools.combinations([None] * nr_of_options + possible_tiles, nr_of_options))
    possible_box_positions = token_mdp.get_box_psoitions()

    print("recording steps:", list(steps_to_record))
    learner = learners.q_learning.QLearning(env=token_mdp, options=[])

    if opt.batch_idx != -1:
        disk_utils.assert_batch_nalloc(opt.batch_idx)
    else:
        opt.batch_idx *= -1

    option_map = options_utils.generate_option_map(learner, token_mdp)

    # do = []
    if use_compelx_options:
        complex_options = tuple('do' + str(act) for act in range(token_mdp.action_space.n))
        option_sets = [o + complex_options for o in option_sets]

    option_sets_scores = {}
    batch_size = len(option_sets) // nr_batches
    batch_start = batch_size * opt.batch_idx
    batch_end = batch_size * (opt.batch_idx + 1)
    batch_data = option_sets[batch_start: batch_end]

    for option_set in tqdm.tqdm(batch_data):
        options = [option_map[goal_idx] for goal_idx in option_set if goal_idx is not None]
        option_sets_scores[option_set] = options_utils.eval_options_on_complex_mdp(
            side_size, options, possible_box_positions, steps_to_record
        )
    return option_sets_scores


def test_options(side_size=7, use_compelx_options=False):
    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=side_size, composite_actions=True)
    # option_set = list(token_mdp.get_walkable_tiles())
    option_set = []
    learner = learners.q_learning.QLearning(env=token_mdp, options=[])
    option_map = options_utils.generate_option_map(learner, token_mdp)

    if use_compelx_options:
        complex_options = tuple('do' + str(act) for act in range(token_mdp.action_space.n))
        option_set.extend(complex_options)
    import time

    for option_id in option_set:
        # option_policy = option_map[option_id]
        option_policy = options_utils.one_step_policy(int(option_id[-1]))
        print(option_id)
        mdp = envs.simple_boxes.BoxWorldSimple(side_size=side_size, composite_actions=True)
        mdp.show_board()
        s = mdp.reset()
        for env_iter in range(100):
            mdp.show_board()

            a = option_policy[s]
            mdp.show_board()
            s, r, d, _ = mdp.step(a)
            mdp.show_board()

            if a != -1:
                print("step", a)
                s, r, d, _ = mdp.step(a)
                mdp.show_board()
                time.sleep(1)
            else:
                print("reset")
                s = mdp.reset()


if __name__ == "__main__":
    # bruteforce_options_complex_world(2, 7, True, True)
    test_options(7, True)
