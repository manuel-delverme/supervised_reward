import numpy as np
import envs.simple_boxes as e
import learners.double_q_learning
import learners.q_learning


def gather_option_stats(SIDE_SIZE, options, possible_box_positions, xs, nr_samples):
    cum_cum_reward = np.zeros(shape=(len(xs), 2))
    for eval_step, box_positions in enumerate(possible_box_positions):
        mean, variance = e.BoxWorldSimple.eval_option_distribution_on_mdp(SIDE_SIZE, box_positions, options, xs, nr_samples)
        cum_cum_reward += np.vstack((mean, variance)).T
    return cum_cum_reward / (eval_step + 1)


def eval_options(SIDE_SIZE, options, possible_box_positions, xs):
    cum_cum_reward = np.zeros(len(xs))
    for eval_step, box_positions in enumerate(possible_box_positions):
        option_set_scores = e.BoxWorldSimple.eval_option_on_mdp(SIDE_SIZE, box_positions, options, xs)
        cum_cum_reward += np.array(option_set_scores)
    return cum_cum_reward / (eval_step + 1)


def select_options(SIDE_SIZE, nr_options, number_of_tiles, reward_vector, sensor_readings):
    scores = []
    for goal_idx, position_idx in enumerate(range(number_of_tiles)):
        sensor_reading = sensor_readings[position_idx]
        score = np.sum(reward_vector[sensor_reading])
        scores.append((goal_idx, score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    options = []
    for goal_idx, goal_score in scores[:nr_options]:
        option = tuple(learners.q_learning.learn_option(goal_idx, e.BoxWorldSimple(side_size=SIDE_SIZE)))
        options.append(option)
        options = sorted(options, key=lambda x: x.index(-1))
    return options


def name_options(options):
    option_names = []
    for option in options:
        for idx, action in enumerate(option):
            if action == -1:
                option_names.append(idx)
                break
    return option_names


def goal_to_policy(learner, o, token_mdp):
    token_mdp.agent_position_idx = o
    learner.generate_option()
    option_vec = tuple(learner.available_actions[-1])
    return option_vec
