import random

import numpy as np
from matplotlib import pyplot as plt

import config
import learners.approx_q_learning
from learners.approx_q_learning import update_reward


def eval_options(env, options):
    cum_score = 0
    for eval_step in range(config.repeat_eval_options):
        _, _, fitness, _ = learners.approx_q_learning.learn(
            environment=env, options=options, training_steps=config.option_eval_training_steps, eval_fitness=True, log_postfix=f'eval_options{eval_step}')
        cum_score += fitness

    return cum_score / (eval_step + 1)


def print_statistics(fitness, options):
    option_names = []
    for option in options:
        option_names.append(int(np.argwhere(option == -1)[0]))
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


def enjoy_policy(environment, policy, reward_function=False):
    environment.render()
    environment.render()
    obs = environment.reset()

    while True:
        print(obs.reshape(-1, config.agent_view_size, config.agent_view_size))
        print('values:',
              *zip(('<', '>', 'foward', 'toggle'), policy.get_value(obs)),
              sep='\n')
        action = policy.get_or_terminate(obs, environment)
        environment.render()

        while action == -1:
            obs = environment.reset()
            action = policy.get_or_terminate(obs, environment)

        obs, reward, terminal, info = environment.step(action)
        if reward_function:
            print('reward:', reward_function(obs, environment))

        cmd = input('q_to_exit, t to terminate, else step')
        if cmd == 'q':
            break
        if cmd == 't':
            obs = environment.reset()


def enjoy_surrogate_reward(environment, surrogate_reward):
    environment.render()
    while True:
        environment.render()
        while True:
            try:
                action = ['a', 'd', 'w', 'e', 'q'].index(input('p'))
            except ValueError:
                pass
            else:
                break

        if action == 4:
            return

        new_state, env_reward, terminal, info = environment.step(action)
        reward, terminal = update_reward(environment, new_state, True, env_reward, terminal, False, surrogate_reward)
        print('updated reward from env', reward, 'real reward', env_reward, 'terminal', terminal)


def plot_surrogate_reward(environment, surrogate_reward):
    print('enjoying reward', surrogate_reward, sep='\n')

    replace_reward = True
    terminate_on_surr_reward = False

    environment.reset()
    rewards = {idx: np.zeros(shape=(environment.env.height, environment.env.width)) for idx in range(4)}
    x1s, x2s, rs = [], [], []
    for _ in range(1000):
        action = random.choice((0, 1, 2, 2, 3))
        # if action == 3:
        #     action = 2

        new_state, reward, terminal, info = environment.step(action)
        reward, terminal = update_reward(environment, new_state, replace_reward, reward, terminal, terminate_on_surr_reward, surrogate_reward)

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
