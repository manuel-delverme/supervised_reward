import numpy as np
import functools
import envs.gridworld
import collections
import tqdm
import sys
import time
import disk_utils
import envs.boxes
import random
import sys

TERMINATE_OPTION = -1


class QLearning(object):
    def __init__(self, env, options=None, epsilon=0.1, gamma=0.90, alpha=0.2, surrogate_reward=None, test_run=False,
                 learning_option=False, train_run=False):

        self.epsilon = epsilon
        self.starting_epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.environment = env
        self.nr_primitive_actions = self.environment.action_space.n
        self.available_actions = list(range(self.environment.action_space.n))

        action_size = env.action_space.n
        self.action_to_id = {k: k for k in range(action_size)}
        if options is not None:
            self.available_actions.extend(options)
            action_size += len(options)
            for option in options:
                # goal = np.argwhere(action == -1)[0]
                self.action_to_id[option] = len(self.action_to_id)
        self.Q = 0.00001 * np.random.rand(env.observation_space.n, action_size)
        self.qmax = np.max(self.Q, axis=1)
        self.qargmax = np.argmax(self.Q, axis=1)

    @functools.lru_cache(maxsize=1024)
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

    def learn(self, max_steps, plot_speed=False):
        cumulative_reward = 0
        time_steps_under_option = 0
        discounted_reward_under_option = 0

        option_begin_state = None
        action_idx = None
        primitive_action = None
        old_state = self.environment.reset()
        new_state = old_state

        steps = range(max_steps)
        if plot_speed:
            import tqdm
            steps = tqdm.tqdm(steps)

        for step in steps:
            action_idx, primitive_action = self.pick_action(old_state, old_action_idx=action_idx,
                                                            old_primitive_action=primitive_action)

            if option_begin_state is None and self.is_option(action_idx):
                option_begin_state = old_state

            if primitive_action != TERMINATE_OPTION:
                new_state, reward, terminal, info = self.environment.step(self.available_actions[primitive_action])
                cumulative_reward += reward

            if primitive_action == TERMINATE_OPTION:
                assert option_begin_state is not None
                time_difference = time_steps_under_option

                qmax = self.qmax[new_state]
                discounted_future_value = (self.gamma ** time_difference) * qmax
                # discounted_future_value = (self.gamma ** time_difference) * np.max(self.Q[new_state, :])
                old_q = self.Q[option_begin_state, action_idx]
                delta_Q = discounted_reward_under_option + discounted_future_value - old_q
                old_q += self.alpha * delta_Q

                if old_q > qmax:
                    self.qmax[old_state] = old_q
                    self.qargmax[old_state] = action_idx

                time_steps_under_option = 0
                option_begin_state = None

            elif self.is_option(action_idx):
                # following option
                time_steps_under_option += 1
                # register the reward following option
                discounted_reward_under_option += reward * (self.gamma ** time_steps_under_option)

                # update the q-value for the last transition off-policy
                # delta_Q = reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[old_state, action_idx]

                old_q = self.Q[old_state, action_idx]
                qmax = self.qmax[new_state]
                delta_Q = reward + self.gamma * qmax - old_q
                old_q += self.alpha * delta_Q

                if old_q > qmax:
                    self.qmax[old_state] = old_q
                    self.qargmax[old_state] = action_idx

            else:
                qmax = np.max(self.Q[new_state, :])
                old_q = self.Q[old_state, action_idx]
                delta_Q = reward + self.gamma * qmax - old_q
                old_q += self.alpha * delta_Q

                if old_q > qmax:
                    self.qmax[old_state] = old_q
                    self.qargmax[old_state] = action_idx

            old_state = new_state
        return

    def test(self, eval_steps):
        cumulative_reward = 0
        primitive_action_idx = None
        action_idx = None

        for tile_idx in range(self.environment.number_of_tiles):
            old_state = self.environment.reset()
            self.environment.teleport_agent(tile_idx)

            for step in range(eval_steps):
                action_idx, primitive_action_idx = self.pick_action_test(
                    old_state, old_primitive_action=primitive_action_idx,
                    old_action_idx=action_idx
                )

                if primitive_action_idx == -1:
                    continue

                new_state, reward, terminal, info = self.environment.step(self.available_actions[primitive_action_idx])
                cumulative_reward += reward

                # self.render_board(render=2)

                old_state = new_state
        return cumulative_reward / self.environment.number_of_tiles

    def is_option(self, action):
        # return action is not None and not isinstance(action, int) and not isinstance(action, np.int64)
        return action is not None and action >= self.nr_primitive_actions


def is_terminate_option(skill, old_state):
    return skill[old_state] == -1


def main():
    import os
    os.chdir("../")
    TEST_MAX_STEPS_EVAL = 1000
    box_positions = (0, 30)

    import envs.simple_boxes
    import learners.double_q_learning
    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=(1, 2))
    learner = learners.double_q_learning.QLearning(env=token_mdp, options=[], test_run=True)
    token_mdp.agent_position_idx = 0
    learner.generate_option()
    option_vec0 = tuple(learner.available_actions[-1])
    token_mdp.agent_position_idx = 17
    learner.generate_option()
    option_vec1 = tuple(learner.available_actions[-1])

    option_vec = [option_vec0, option_vec1]

    option_set_score = {}
    mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=box_positions)
    learner = QLearning(env=mdp, options=option_vec, test_run=True)
    import tqdm

    training_time = 0
    testing_time = 0
    # for _ in tqdm.tqdm(range(100)):
    for it in range(10):
        print(it)
        learner = QLearning(env=mdp, options=option_vec, test_run=False)
        time0 = time.time()
        learner.learn(max_steps=10000, plot_speed=True)
        diff = (time.time() - time0)
        training_time += diff

        time0 = time.time()
        cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
        diff = (time.time() - time0)
        testing_time += diff
        # cum_cum_reward += cum_reward
    print("training_time", training_time, "testing_time", testing_time, "train/test", float(training_time) / testing_time)


if __name__ == "__main__":
    main()
