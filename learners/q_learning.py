import random
import time

import numpy as np

import config
from utils import disk_utils

TERMINATE_OPTION = -1


class QLearning(object):
    def __init__(self, env, options=None, epsilon=0.1, gamma=0.90, alpha=0.2, surrogate_reward=None, goal=None):
        self.goal = goal
        self.epsilon = epsilon
        self.starting_epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.environment = env
        self.nr_primitive_actions = self.environment.action_space.n
        self.available_actions = list(range(self.nr_primitive_actions))

        action_size = env.action_space.n
        self.action_to_id = {k: k for k in range(action_size)}
        if options is not None:
            self.available_actions.extend(options)
            action_size += len(options)
            for option in options:
                self.action_to_id[option] = len(self.action_to_id)
        self.Q = 0.00001 * np.random.rand(env.observation_space.n, action_size)
        self.qmax = np.max(self.Q, axis=1)
        self.qargmax = np.argmax(self.Q, axis=1)
        self.surrogate_reward = surrogate_reward

    def pick_action_test(self, state, old_action_idx, old_primitive_action):
        return self.pick_action(state, old_action_idx, old_primitive_action, explore=True)

    def pick_action(self, state, old_action_idx, old_primitive_action, explore=True):
        # if the option terminated OR i was using primitives
        if not self.is_option(old_action_idx) or old_primitive_action == TERMINATE_OPTION:
            if not explore or self.epsilon < random.random():
                action_idx = self.qargmax[state]
                # print("policy pick", action_idx)
            else:
                action_idx = random.randint(0, self.nr_primitive_actions - 1)
                while self.is_option(action_idx) and self.available_actions[action_idx][state] == TERMINATE_OPTION:
                    action_idx = random.randint(0, self.nr_primitive_actions - 1)

            if self.is_option(action_idx):
                primitive_action_idx = self.available_actions[action_idx][state]
            else:
                primitive_action_idx = action_idx
        else:
            # keep following the option
            assert self.is_option(old_action_idx)
            primitive_action_idx = self.available_actions[old_action_idx][state]
            action_idx = old_action_idx

        return action_idx, primitive_action_idx

    def learn(self, generate_options=False, plot_every=None, xs=None, replace_reward=config.learn.replace_reward,
              generate_on_rw=config.learn.generate_on_rw, use_learned_options=config.learn.use_learned_options):

        xs = set(xs)

        cumulative_reward = 0
        fitness = 0
        time_steps_under_option = 0
        discounted_reward_under_option = 0

        option_begin_state = None
        action_idx = None
        primitive_action = None
        old_state = self.environment.reset()
        new_state = old_state
        option_goals = set()
        fitnesses = []

        terminal = False
        reward = None
        render_one = False

        for step in range(max(xs)):
            action_idx, primitive_action = self.pick_action(old_state, old_action_idx=action_idx,
                                                            old_primitive_action=primitive_action)

            if option_begin_state is None and self.is_option(action_idx):
                option_begin_state = old_state

            if primitive_action != TERMINATE_OPTION:
                new_state, reward, terminal, info = self.environment.step(self.available_actions[primitive_action])
                if reward > 0:
                    fitness += 1
                if self.surrogate_reward is not None:
                    if replace_reward:
                        reward = self.surrogate_reward(new_state)
                        terminal = reward > 0
                    else:
                        terminal = reward > 0
                        reward += self.surrogate_reward(new_state)
                cumulative_reward += reward

            future_value = self.qmax[new_state]

            if plot_every is not None and step % plot_every == 0 and step > 0:
                render_one = True

            if terminal:
                self.environment.reset()
                if self.is_option(action_idx):
                    primitive_action = TERMINATE_OPTION

                if render_one:
                    # self.render_board(render=2, info={'step': step}, highlight_square=self.goal)
                    print('rendering at', step)
                    self.test(eval_steps=200, render=True)
                    render_one = False

            if primitive_action == TERMINATE_OPTION:
                time_difference = time_steps_under_option
                old_state = option_begin_state
                discounted_reward = discounted_reward_under_option
            else:
                time_difference = 1
                discounted_reward = reward
                if self.is_option(action_idx):
                    assert option_begin_state is not None
                    time_steps_under_option += 1
                    discounted_reward_under_option += reward * (self.gamma ** time_steps_under_option)

            discounted_future_value = (self.gamma ** time_difference) * future_value

            old_q = self.Q[old_state, action_idx]
            delta_Q = discounted_reward + discounted_future_value - old_q
            new_q = old_q + self.alpha * delta_Q

            self.Q[old_state, action_idx] = new_q

            if primitive_action == TERMINATE_OPTION:
                time_steps_under_option = 0
                option_begin_state = None

            # found a better max
            if new_q > self.qmax[old_state]:
                self.qmax[old_state] = new_q
                self.qargmax[old_state] = action_idx

            # the max was updated
            elif action_idx == self.qargmax[old_state]:
                arg_max = np.argmax(self.Q[old_state])
                self.qargmax[old_state] = arg_max
                self.qmax[old_state] = self.Q[old_state][arg_max]

            if generate_on_rw:
                option_gen_metric = reward
            else:
                option_gen_metric = delta_Q

            if generate_options and option_gen_metric > 0 and self.environment.agent_position_idx not in option_goals:
                # reward = self.surrogate_reward(self.environment)
                # goal = self.environment.agent_position_idx
                option_goals.add(self.generate_option(old_state, use_learned_options))

            old_state = new_state

            if step in xs:
                test_fitness = self.test(step)
                fitnesses.append(test_fitness)
        fitnesses.append(fitness)

        opts = self.available_actions[self.environment.action_space.n:]
        return opts, cumulative_reward, fitnesses

    def generate_option(self, goal, use_learned_options):
        new_option = learn_option(goal, self.environment)
        new_option = tuple(new_option)

        self.available_actions.append(new_option)
        if use_learned_options:
            # new_option = learn_option(old_state, self.environment)
            option_idx = self.Q.shape[1] + 1
            self.action_to_id[new_option] = option_idx - 1
            tmp_Q = np.empty((self.Q.shape[0], option_idx))
            tmp_Q[:, :-1] = self.Q
            self.Q = tmp_Q
            self.Q[:, -1] = self.Q[:, :-1].mean(axis=1)
        return self.environment.agent_position_idx

    def test(self, eval_steps, render=False):
        cumulative_reward = 0
        fitness = 0
        primitive_action_idx = None
        action_idx = None

        for tile_idx in range(1 if render else 100):
            old_state = self.environment.reset()
            # self.environment.teleport_agent(tile_idx)

            for step in range(eval_steps):
                action_idx, primitive_action_idx = self.pick_action_test(
                    old_state, old_primitive_action=primitive_action_idx,
                    old_action_idx=action_idx
                )

                if primitive_action_idx == -1:
                    continue

                new_state, reward, terminal, info = self.environment.step(self.available_actions[primitive_action_idx])
                cumulative_reward += reward
                fitness += 1 if reward > 0 else 0

                if terminal:
                    self.environment.reset()

                if render:
                    if action_idx > self.environment.action_space.n:
                        goal = self.available_actions[action_idx].index(-1)
                    else:
                        goal = None
                    self.render_board(render=2, info={'reward': cumulative_reward}, highlight_square=goal)

                old_state = new_state
        return fitness / 100

    def is_option(self, action):
        # return action is not None and not isinstance(action, int) and not isinstance(action, np.int64)
        return action is not None and action >= self.nr_primitive_actions

    def render_board(self, render, highlight_square=None, sleep_time=1. / 30., info={}):
        if render > 0:
            render -= 1
            time.sleep(sleep_time)
            self.environment.show_board(
                some_matrix=self.qmax,
                policy=self.qargmax,
                highlight_square=highlight_square,
                info=info,
                option_vec=self.available_actions[6:]
            )
            if render == 0:
                input("sleeping")
        return render


def is_terminate_option(skill, old_state):
    return skill[old_state] == -1


@disk_utils.disk_cache
# @gin.configurable
def learn_option(goal, mdp, training_steps=10000):  # reduced for 7x7
    print("generating policy for goal:{}".format(goal))

    def surrogate_reward(state):
        if goal == state:
            return 100
        else:
            return -1

    learner = QLearning(env=mdp, options=None, surrogate_reward=surrogate_reward, goal=goal)
    _, _, fitnesses = learner.learn(xs=[training_steps, ], plot_every=None)  # plot_every=training_steps//5)
    option = np.argmax(learner.Q, axis=1)

    if hasattr(mdp, 'number_of_tiles'):
        state_idx = goal
        try:
            while True:
                option[state_idx] = -1
                state_idx += mdp.number_of_tiles
        except IndexError as e:
            pass

        option = np.tile(
            option, mdp.observation_space.n // mdp.number_of_tiles
        )
    else:
        option[goal] = -1
    return option
