import numpy as np
import disk_utils
import heapq
import random


class DoubleQLearning(object):
    def __init__(self, env, options=None, epsilon=0.1, gamma=0.99, alpha=0.1, surrogate_reward=None):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.Q1 = 0.00001 * np.random.rand(env.observation_space.n, env.action_space.n)
        self.Q2 = 0.00001 * np.random.rand(env.observation_space.n, env.action_space.n)
        if hasattr(env, 'terminal_states'):
            for state in env.terminal_states:
                self.Q1[state, :] = 0
                self.Q2[state, :] = 0
        self.previous_action = None
        self.environment = env
        self.surrogate_reward = surrogate_reward
        self.available_actions = list(range(self.environment.action_space.n))
        if options is not None:
            self.available_actions.extend(options)

    @staticmethod
    def null_criterion(*args, **kwargs):
        return False

    def pick_action(self, old_state):
        TERMINATE_OPTION = -1
        # i was following an option and should still follow it
        if is_option(self.previous_action) and not self.previous_action[old_state] == TERMINATE_OPTION:
            # keep going
            return self.previous_action, self.previous_action[old_state]

        # if the option terminated OR i was using primitives
        # Q_exploit
        primitive_action = TERMINATE_OPTION
        # epsilon could be silly and choose a terminated option
        # pick again is faster than checking for initiation sets
        while primitive_action == TERMINATE_OPTION:
            if self.epsilon > random.random():
                Q_total_s = self.Q1[old_state, :] + self.Q2[old_state, :]
                best_action = np.argmax(Q_total_s)
                action = self.available_actions[best_action]
            else:
                action = random.choice(self.available_actions)

            if is_option(action):
                primitive_action = action[old_state]
            else:
                primitive_action = action

        self.previous_action = action
        return action, primitive_action

    def learn(self, training_steps, goal_criterion=None):
        # returnSum = 0.0
        # print("[double_q] learn")
        discovered_options = []
        cumulative_reward = 0

        old_state = self.environment.reset()
        time_steps_under_option = 0
        for episode_num in range(training_steps):
            action, primitive_action = self.pick_action(old_state)
            new_state, reward, terminal, info = self.environment.step(primitive_action)

            if self.surrogate_reward is not None:
                reward = self.surrogate_reward(self.environment)
            cumulative_reward += reward

            if is_option(action):
                time_steps_under_option += 1
            else:
                if random.random() > 0.5:
                    Q, q = self.Q1, self.Q2
                else:
                    Q, q = self.Q2, self.Q1

                # TODO: off by one in Q values? should i consider the past transition that led me here instead
                #  of the outgoing one?
                best_future_q = np.max(Q[new_state, :])
                old_Q = Q[old_state, action]
                k = 1 + time_steps_under_option
                delta_Q = reward + ((self.gamma ** k) * best_future_q) - old_Q

                # update the value in self.Q1 or self.Q2 by pointer
                old_Q += self.alpha * delta_Q

                # TODO: or new state?
                if goal_criterion is not None and goal_criterion(old_state, delta_Q):
                    new_option = learn_option(old_state, self.environment)
                    self.available_actions.append(new_option)
                    # TODO: preallocate the matrix
                    # TODO: reinit last row ~as what?~
                    option_idx = self.Q1.shape[1] + 1
                    tmp_Q1 = np.empty((self.Q1.shape[0], option_idx))
                    tmp_Q1[:, :-1] = self.Q1
                    self.Q1 = tmp_Q1
                    self.Q1[:, -1].fill(0)

                    tmp_Q2 = np.empty((self.Q2.shape[0], option_idx))
                    tmp_Q2[:, :-1] = self.Q2
                    self.Q2 = tmp_Q2
                    self.Q2[:, -1].fill(0)

                old_state = new_state
                # cumulative_reward += (self.time_limit - timestep) * reward
        return self.available_actions[self.environment.action_space.n:], cumulative_reward


def is_option(action):
    # return isinstance(action, np.ndarray)
    return action is not None and not isinstance(action, int) and not isinstance(action, np.int64)


def is_terminate_option(skill, old_state):
    return skill[old_state] == -1


@disk_utils.disk_cache
def learn_option(goal, mdp):
    # print("generating policy for goal:", goal)

    def surrogate_reward(_mdp):
        return 1 if goal == _mdp.agent_position_idx else -1

    learner = DoubleQLearning(
        env=mdp,
        options=None,
        epsilon=0.1,
        gamma=0.99,
        alpha=0.1,
        surrogate_reward=surrogate_reward
    )
    _, _ = learner.learn(training_steps=10000)
    option = np.argmax(learner.Q1 + learner.Q2, axis=1)
    option[goal] = -1
    return option
