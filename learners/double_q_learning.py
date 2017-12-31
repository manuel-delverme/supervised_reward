import numpy as np
import collections
import tqdm
import learners.policy_iter
import sys
import time
import disk_utils
import envs.boxes
import random
import sys


class QLearning(object):
    def __init__(
            self, env, options=None, epsilon=0.1, gamma=0.90, alpha=0.3, surrogate_reward=None, test_run=False,
            learning_option=False, train_run=False):
        self.epsilon = epsilon
        self.starting_epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        # self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.previous_action = None
        self.environment = env
        self.surrogate_reward = surrogate_reward
        self.available_actions = list(range(self.environment.action_space.n))

        assert (train_run ^ test_run ^ learning_option)
        self.test_run = test_run
        self.train_run = train_run
        self.learning_option = learning_option

        action_size = env.action_space.n
        if options is not None:
            self.available_actions.extend(options)
            action_size += len(options)

        self.Q1 = 0.00001 * np.random.rand(env.observation_space.n, action_size)
        self.Q2 = 0.00001 * np.random.rand(env.observation_space.n, action_size)

    def pick_action(self, old_state, kill_option=False):
        TERMINATE_OPTION = -1
        # i was following an option and should still follow it
        if (is_option(self.previous_action) and not self.previous_action[old_state] == TERMINATE_OPTION) and not kill_option:
            # keep going
            # return self.previous_action, self.previous_action[old_state]
            return self.previous_action, self.previous_action[old_state]

        # if the option terminated OR i was using primitives
        # Q_exploit
        primitive_action = TERMINATE_OPTION
        # epsilon could be silly and choose a terminated option
        # pick again is faster than checking for initiation sets
        while primitive_action == TERMINATE_OPTION:
            if self.epsilon < random.random():
                best_action = np.argmax(self.Q1[old_state] + self.Q2[old_state])
                action = self.available_actions[best_action]
            else:
                action = random.choice(self.available_actions)

            if is_option(action):
                primitive_action = action[old_state]
            else:
                primitive_action = action
        self.previous_action = action
        return action, primitive_action

    def learn(self, steps_of_no_change=None, generate_options=False, max_steps=None, plot_progress=True):
        assert (steps_of_no_change is not None or max_steps is not None)

        if plot_progress:
            if steps_of_no_change is not None:
                progress_bar = tqdm.tqdm(total=steps_of_no_change, file=sys.stdout)
            elif self.train_run or self.test_run :
                progress_bar = tqdm.tqdm(total=max_steps, file=sys.stdout)

        cumulative_reward = 0
        terminal = True
        time_steps_under_option = 0
        no_change = 0
        max_no_change = 0
        render = 0
        option_goals = set()
        old_states = collections.deque(range(20), maxlen=20)

        for step in range(max_steps):
            if terminal:
                old_state = self.environment.reset()

            old_states.append(old_state)
            kill_option = not len(set(old_states)) > 4
            action, primitive_action = self.pick_action(old_state, kill_option)

            # if old_pick == primitive_action:
            #     if is_option(action) and self.test_run:
            #         print("picked option", time_steps_under_option, envs.boxes.BoxWorldActions(primitive_action))
            #         self.environment.print_board(policy=action)
            #         print("uffa")
            # else:
            #     old_pick = primitive_action

            new_state, reward, terminal, info = self.environment.step(primitive_action)

            if self.surrogate_reward is not None:
                reward = self.surrogate_reward(self.environment)
                if reward > 0 and self.learning_option:
                    terminal = True

            cumulative_reward += reward

            if is_option(action) and (self.previous_action == action).all():
                time_steps_under_option += 1
            else:
                Q, q = (self.Q1, self.Q2) if random.random() > 0.5 else (self.Q2, self.Q1)

                # old_choice = np.argmax(q[old_state] + Q[old_state])
                delta_Q = reward + ((self.gamma ** (1 + time_steps_under_option)) * np.max(q[new_state, :])) - Q[
                    old_state, action]
                Q[old_state][action] += self.alpha * delta_Q
                time_steps_under_option = 0
                # progress_bar.set_description(str(delta_Q))
                # new_choice = np.argmax(q[old_state] + Q[old_state])

                if abs(delta_Q) < 0.01:
                    no_change += 1
                else:
                    no_change = 0

                # render = self.render_board(step, render)
                if generate_options and delta_Q > 1 and self.environment.agent_position_idx not in option_goals:
                    option_goals.add(self.generate_option())

                if steps_of_no_change is not None and no_change > max_no_change:
                    max_no_change = no_change
                    if plot_progress and self.learning_option:
                        progress_bar.update(1)
            old_state = new_state

            if plot_progress and (self.train_run or self.test_run):
                progress_bar.update(1)

            if steps_of_no_change is not None and no_change > steps_of_no_change:
                break
        else:
            pass
            # print("max step, break")
        return self.available_actions[self.environment.action_space.n:], cumulative_reward

    def render_board(self, steps, render):
        if steps % 5000 == 0:
            render = 50

        if render > 0:
            render -= 1
            time.sleep(1 / 30)
            self.environment.print_board(
                some_matrix=np.max(self.Q1 + self.Q2, axis=1),
                policy=np.argmax(self.Q1 + self.Q2, axis=1),
            )
        return render

    def generate_option(self):
        new_option = learn_option(self.environment.agent_position_idx, self.environment)
        # new_option = learn_option(old_state, self.environment)
        self.available_actions.append(new_option)
        option_idx = self.Q1.shape[1] + 1
        tmp_Q = np.empty((self.Q1.shape[0], option_idx))
        tmp_Q[:, :-1] = self.Q1
        self.Q1 = tmp_Q
        self.Q1[:, -1] = self.Q1.mean(axis=1)
        tmp_Q = np.empty((self.Q2.shape[0], option_idx))
        tmp_Q[:, :-1] = self.Q2
        self.Q2 = tmp_Q
        self.Q2[:, -1] = self.Q2.mean(axis=1)
        return self.environment.agent_position_idx


def is_option(action):
    # return isinstance(action, np.ndarray)
    return action is not None and not isinstance(action, int) and not isinstance(action, np.int64)


def is_terminate_option(skill, old_state):
    return skill[old_state] == -1


@disk_utils.disk_cache
def learn_option(goal, mdp):
    print("\ngenerating policy for goal:{}\n".format(goal))

    def surrogate_reward(_mdp):
        # return 1 if goal == _mdp._hash_state() else -1
        return 10 if goal == _mdp.agent_position_idx else -1

    # learner = learners.policy_iter.PolicyIteration(
    #     env=mdp,
    #     options=None,
    #     epsilon=0.1,
    #     gamma=0.90,
    #     alpha=0.1,
    #     surrogate_reward=surrogate_reward,
    # )

    learner = QLearning(
        env=mdp, options=None, epsilon=0.1, gamma=0.90, alpha=0.1, surrogate_reward=surrogate_reward,
        learning_option=True
    )
    _ = learner.learn(steps_of_no_change=10, max_steps=100000)
    option = np.argmax(learner.Q1 + learner.Q2, axis=1)

    state_idx = goal
    try:
        while True:
            state_idx += mdp.number_of_tiles
            option[state_idx] = -1
    except IndexError as e:
        pass
    mdp.print_board(
        some_matrix=np.max(learner.Q1 + learner.Q2, axis=1),
        policy=option,
    )
    print("done")
    # time.sleep(1000)
    return option
