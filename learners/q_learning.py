import random
import gin
import time
import tqdm

import numpy as np
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

    def learn(self, generate_options=False, plot_every=None, xs=None, plot_progress=False):
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

        for step in range(max(xs)):
            if plot_every is not None and step % plot_every == 0:
                self.render_board(render=2, info={'step': step}, highlight_square=self.goal)

            action_idx, primitive_action = self.pick_action(old_state, old_action_idx=action_idx,
                                                            old_primitive_action=primitive_action)

            if option_begin_state is None and self.is_option(action_idx):
                option_begin_state = old_state

            if primitive_action != TERMINATE_OPTION:
                new_state, reward, terminal, info = self.environment.step(self.available_actions[primitive_action])
                if self.surrogate_reward is not None:
                    reward = self.surrogate_reward(new_state)
                    terminal = reward > 0
                else:
                    if reward > 0:
                        fitness += 1
                cumulative_reward += reward

            future_value = self.qmax[new_state]

            if terminal:
                self.environment.reset()
                if self.is_option(action_idx):
                    primitive_action = TERMINATE_OPTION

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

            if generate_options and delta_Q > 0 and self.environment.agent_position_idx not in option_goals:
                # reward = self.surrogate_reward(self.environment)
                # goal = self.environment.agent_position_idx
                option_goals.add(self.generate_option(old_state))

            old_state = new_state

            if step in xs:
                test_fitness = self.test(step)
                fitnesses.append(test_fitness)
        fitnesses.append(fitness)

        opts = self.available_actions[self.environment.action_space.n:]
        return opts, cumulative_reward, fitnesses

    def generate_option(self, goal):
        new_option = learn_option(goal, self.environment)
        new_option = tuple(new_option)

        # new_option = learn_option(old_state, self.environment)
        self.available_actions.append(new_option)
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

        for tile_idx in range(self.environment.number_of_tiles):
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
                    if action_idx > 3:
                        goal = self.available_actions[action_idx].index(-1)
                    else:
                        goal = None
                    self.render_board(render=2, info={'reward': cumulative_reward}, highlight_square=goal)

                old_state = new_state
        # return cumulative_reward / self.environment.number_of_tiles
        return fitness / self.environment.number_of_tiles

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


def main():
    import os
    os.chdir("../")
    TEST_MAX_STEPS_EVAL = 1000
    box_positions = (0, 30)

    import envs.simple_boxes
    import learners.DISABLED_double_q
    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=(1, 2))
    learner = learners.DISABLED_double_q.QLearning(env=token_mdp, options=[], test_run=True)
    token_mdp.agent_position_idx = 0
    learner.generate_option()
    option_vec0 = tuple(learner.available_actions[-1])
    token_mdp.agent_position_idx = 17
    learner.generate_option()
    option_vec1 = tuple(learner.available_actions[-1])

    option_vec = [option_vec0, option_vec1]

    mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=box_positions)
    learner = QLearning(env=mdp, options=option_vec)

    training_time = 0
    testing_time = 0
    for it in range(10):
        print(it)
        learner = QLearning(env=mdp, options=option_vec, test_run=False)
        time0 = time.time()
        learner.learn(xs=[10000,] , plot_speed=True)
        diff = (time.time() - time0)
        training_time += diff

        time0 = time.time()
        cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
        diff = (time.time() - time0)
        testing_time += diff
        # cum_cum_reward += cum_reward
    print("training_time", training_time, "testing_time", testing_time, "train/test",
          float(training_time) / testing_time)


@disk_utils.disk_cache
# @gin.configurable
def learn_option(goal, mdp, training_steps=100000):
    print("generating policy for goal:{}".format(goal))

    def surrogate_reward(state):
        if goal == state:
            return 100
        else:
            return -1

    learner = QLearning(env=mdp, options=None, surrogate_reward=surrogate_reward, goal=goal)
    _, _, fitnesses = learner.learn(xs=[training_steps, ], plot_every=None) # training_steps//10)
    option = np.argmax(learner.Q, axis=1)

    state_idx = goal
    try:
        while True:
            option[state_idx] = -1
            state_idx += mdp.number_of_tiles
    except IndexError as e:
        pass

    # mdp.show_board(
    #     some_matrix=np.max(learner.Q, axis=1),
    #     # some_matrix=value,
    #     policy=np.array(option),
    # )
    # input("done")
    option = np.tile(
        option, mdp.observation_space.n // mdp.number_of_tiles
    )
    # time.sleep(1000)
    return option


if __name__ == "__main__":
    main()
