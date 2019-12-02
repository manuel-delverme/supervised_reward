import random
import shared.constants
import time

import numpy as np
from matplotlib import pyplot as plt

import config
import learners.approx_q_learning
import learners.approx_q_learning


def eval_options(env, options):
    print('eval options', options)
    cum_score = 0
    for eval_step in range(config.repeat_eval_options):
        _, _, fitness, _ = learners.approx_q_learning.learn(
            environment=env, options=options, training_steps=config.option_eval_training_steps, eval_fitness=True, log_postfix=f'eval_options{eval_step}')
        cum_score += fitness

    return cum_score / (eval_step + 1)


def print_statistics(fitness, options):
    option_names = []
    for option in options:
        option_names.append(int(np.argwhere(option == shared.constants.TERMINATE_OPTION)[0]))
    option_names = " ".join(str(n) for n in sorted(option_names))
    print("score:\t{}\toptions: {}\t{}".format(fitness, len(options), option_names))


def to_tuple(img):
    b = tuple(tuple(tuple(column) for column in row) for row in img)
    return b


def plot_intinsic_motivation(motivating_function, old_state, step):
    plottable_reward = motivating_function[:-1].reshape(old_state)
    for layer in range(plottable_reward.shape[2]):
        figure, axes = plt.figure(figsize=plottable_reward.shape[:-1], dpi=1), plt.gca()
        axes.matshow(plottable_reward[:, :, layer], vmax=2, vmin=-2, interpolation='nearest')
        axes.axis('off')
        config.tensorboard.add_figure(f'motivating_functions/{layer}', figure, global_step=step, close=True)


def enjoy_policy(environment, policy, available_actions, reward_function=False, type_of_run=None):
    forced_action = None
    environment.render()
    environment.render()
    obs = environment.reset()
    try:
        policy.reward_function.reset()
    except AttributeError:
        pass

    while True:
        for idx, (m, n, v) in enumerate(zip(policy.estimator.models, ('<', '>', 'foward', 'toggle', *range(10)), policy.get_value(obs))):
            print(n, v)

        active_policy = policy
        while hasattr(active_policy, 'get_or_terminate'):
            action_idx = active_policy.get_or_terminate(obs, environment)
            if action_idx is not None:
                active_policy = available_actions[action_idx]
            else:
                active_policy = None
                break

        if forced_action is not None:
            action_idx = forced_action
            forced_action = None
        else:
            action_idx = active_policy

        while action_idx == shared.constants.TERMINATE_OPTION:
            obs = environment.reset()
            policy.motivating_function.reset()

            time.sleep(1)
            action_idx = policy.get_or_terminate(obs, environment)
            print('a', action_idx, 'r', policy.motivating_function(obs, environment))
            if action_idx == shared.constants.TERMINATE_OPTION:
                print('degenerate policy, terminates on reset')
                return

        obs, env_reward, terminal, info = environment.step(action_idx)

        if reward_function:
            # reward = reward_function(obs, environment)
            from learners.approx_q_learning import update_reward
            reward, terminal = update_reward(environment, obs, replace_reward=True, reward=env_reward, steps_since_last_restart=-1, terminal=terminal, surrogate_reward=reward_function, type_of_run=type_of_run)
            print('reward:', reward)
            if reward == config.option_termination_treshold:
                print('forcing terminal')
                terminal = True
                policy.motivating_function.reset()

        environment.render(observation=obs)
        cmd = input('q to exit, t to terminate')
        if cmd == 'q':
            break
        elif cmd == 't' or terminal:
            obs = environment.reset()
            policy.motivating_function.reset()
        else:
            try:
                forced_action = ['a', 'd', 'w', 'e', '1', '2'].index(cmd)
            except ValueError:
                pass


def enjoy_surrogate_reward(environment, surrogate_reward, inibited_rewards=(), type_of_run="visualization"):
    from learners.approx_q_learning import update_reward
    obs = environment.reset()
    environment.render(observation=obs)
    environment.render(observation=obs)
    while True:
        while True:
            try:
                action = ['a', 'd', 'w', 'e', 'q'].index(input('take an action:'))
            except ValueError:
                pass
            else:
                break

        if action == 4:
            return

        new_state, env_reward, terminal, info = environment.step(action)
        reward, terminal = update_reward(environment, new_state, replace_reward=True, reward=env_reward, steps_since_last_restart=-1, terminal=terminal,
                                         surrogate_reward=surrogate_reward, type_of_run=type_of_run)

        with np.printoptions(precision=3, suppress=True):
            print('updated reward from env', reward, 'real reward', env_reward, 'terminal', terminal)
        environment.render(observation=new_state)
        if terminal:
            environment.reset()
            surrogate_reward.reset()

        if surrogate_reward.ltl_progress == len(surrogate_reward.target_state):
            surrogate_reward.reset()


def plot_surrogate_reward(environment, surrogate_reward):
    from learners.approx_q_learning import update_reward
    with np.printoptions(precision=3, suppress=True):
        print('enjoying reward', surrogate_reward, sep='\n')

    replace_reward = True

    environment.reset()
    rewards = {idx: np.zeros(shape=(environment.env.height, environment.env.width)) for idx in range(4)}
    x1s, x2s, rs = [], [], []
    for _ in range(1000):
        action = random.choice((0, 1, 2, 2, 3))
        # if action == 3:
        #     action = 2

        new_state, reward, terminal, info = environment.step(action)
        reward, terminal = update_reward(environment, new_state, replace_reward, reward, steps_since_last_restart=-1, surrogate_reward=surrogate_reward,
                                         terminal=terminal, type_of_run='vizualization')

        if terminal:
            environment.reset()

        x1, x2 = environment.env.agent_pos
        x1s.append(x1)
        x2s.append(x2)
        rs.append(reward)

        rewards[environment.env.agent_dir][x1, x2] = reward

    # fig = plt.figure(figsize=(2, 2))
    # fig, axs = plt.subplots(2, 2)
    # plt.subplot(2, 2, 2, frameon=False)

    for idx, (k, v) in enumerate(rewards.items()):
        v = v[min(x1s): max(x1s) + 1, min(x2s): max(x2s) + 1]

        plt.subplot(2, 2, idx + 1)
        plt.title(
            ['right', 'down', 'left', 'up'][k]
        )
        plt.imshow(v.transpose(), vmin=min(rs), vmax=max(rs))
        cbar = plt.colorbar()
        # plt.legend()
    plt.show()


def hash_image(image):
    return image.data.tobytes()
