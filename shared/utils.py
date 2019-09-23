import random
import learners.approx_q_learning
import time

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
    try:
        policy.reward_function.reset()
    except AttributeError:
        pass
    inspect = False

    while True:
        # print(obs.reshape(-1, config.agent_view_size, config.agent_view_size))
        dt = []
        for idx, (m, n, v) in enumerate(zip(policy.estimator.models, ('<', '>', 'foward', 'toggle'), policy.get_value(obs))):
            if inspect:
                w = m.regressor.weight.reshape(-1).detach().numpy()
                w = np.multiply(w, obs.reshape(-1))
                w = w.reshape(obs.shape)  # .transpose(1, 2, 0)

                plt.suptitle(f'weights for action {n}')
                for layer in range(w.shape[0]):
                    plt.subplot(2, 2, layer + 1)
                    plt.imshow(w[layer, :, :], vmin=w.min(), vmax=w.max())
                    cbar = plt.colorbar()
                plt.show()
                b = m.regressor.bias.detach().numpy()
            else:
                b = None

            dt.append((n, b, v))
        plt.show()

        print(*dt, sep='\n')

        action = policy.get_or_terminate(obs, environment)

        while action == -1:
            obs = environment.reset()
            policy.reward_function.reset()

            time.sleep(1)
            action = policy.get_or_terminate(obs, environment)
            print('a', action, 'r', policy.reward_function)

        obs, reward, terminal, info = environment.step(action)

        if reward_function:
            reward = reward_function(obs, environment)
            print('reward:', reward)
            if reward == config.option_termination_treshold:
                print('forcing terminal')
                terminal = True
                policy.reward_function.reset()

        environment.render(observation=obs)
        inspect = False
        cmd = input('q_to_exit, t to terminate, i for inspect else step')
        if cmd == 'q':
            break
        if cmd == 't' or terminal:
            obs = environment.reset()
            policy.reward_function.reset()
        if cmd == 'i':
            inspect = True


def enjoy_surrogate_reward(environment, surrogate_reward):
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
                                         surrogate_reward=surrogate_reward, type_of_run='visualization')

        with np.printoptions(precision=3, suppress=True):
            print('updated reward from env', reward, 'real reward', env_reward, 'terminal', terminal)
        environment.render(observation=new_state)


def plot_surrogate_reward(environment, surrogate_reward):
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
