import collections
import sys
import time
import math
import numpy as np
import random


class QLearning(object):
    def __init__(self, env, options=None, epsilon=0.1, gamma=0.90, alpha=0.1, surrogate_reward=None):
        self.epsilon = epsilon
        self.starting_epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        # self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        self.Q1 = 0.00001 * np.random.rand(env.observation_space.n, env.action_space.n)
        self.Q2 = 0.00001 * np.random.rand(env.observation_space.n, env.action_space.n)
        self.previous_action = None
        self.environment = env
        self.surrogate_reward = surrogate_reward
        self.available_actions = list(range(self.environment.action_space.n))
        if options is not None:
            self.available_actions.extend(options)

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

    def learn(self, steps_of_no_change=sys.maxsize, generate_options=False, max_steps=sys.maxsize):
        assert (steps_of_no_change != sys.maxsize or max_steps != sys.maxsize)
        cumulative_reward = 0
        terminal = True
        time_steps_under_option = 0
        no_change = 0
        steps = 0
        max_no_change = 0
        render = 0

        for step in range(max_steps):
            if no_change > steps_of_no_change:
                break

            if terminal:
                old_state = self.environment.reset()

            action, primitive_action = self.pick_action(old_state)
            steps += 1
            new_state, reward, terminal, info = self.environment.step(primitive_action)

            if self.surrogate_reward is not None:
                reward = self.surrogate_reward(self.environment)
            else:
                if reward == 1:
                    terminal = True
            cumulative_reward += reward

            if is_option(action):
                time_steps_under_option += 1
            else:
                if random.random() > 0.5:
                    Q, q = self.Q1, self.Q2
                else:
                    Q, q = self.Q2, self.Q1

                old_choice = np.argmax(q[old_state] + Q[old_state])

                best_future_q = np.max(q[new_state, :])
                old_Q = Q[old_state, action]
                k = 1 + time_steps_under_option
                delta_Q = reward + ((self.gamma ** k) * best_future_q) - old_Q
                Q[old_state][action] += self.alpha * delta_Q

                new_choice = np.argmax(q[old_state] + Q[old_state])

                if old_choice == new_choice:
                    no_change += 1
                else:
                    no_change = 0
                if no_change > max_no_change and steps_of_no_change < max_steps:
                    self.epsilon = self.starting_epsilon * (1 - max_no_change / steps_of_no_change)
                    max_no_change = no_change
                    print(steps, no_change)

                if steps % 100000 == 0:
                    render = 50

                if render > 0:
                    render -= 1
                    time.sleep(1 / 30)
                    self.environment.render_board(
                        some_matrix=np.max(self.Q1 + self.Q2, axis=1),
                        policy=np.argmax(self.Q1 + self.Q2, axis=1),
                    )

                # TODO: or new state?
                if generate_options and delta_Q > 1:
                    new_option = learn_option(old_state, self.environment)
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

                old_state = new_state
        self.environment.render_board(
            some_matrix=np.max(self.Q1 + self.Q2, axis=1),
            policy=np.argmax(self.Q1 + self.Q2, axis=1),
        )
        time.sleep(5)
        return self.available_actions[self.environment.action_space.n:], cumulative_reward


def is_option(action):
    # return isinstance(action, np.ndarray)
    return action is not None and not isinstance(action, int) and not isinstance(action, np.int64)


def is_terminate_option(skill, old_state):
    return skill[old_state] == -1


# @disk_utils.disk_cache
def learn_option(goal, mdp):
    print("generating policy for goal:", goal)

    def surrogate_reward(_mdp):
        return 1 if goal == _mdp._hash_state() else -1

    learner = QLearning(
        env=mdp,
        options=None,
        epsilon=0.1,
        gamma=0.90,
        alpha=0.1,
        surrogate_reward=surrogate_reward
    )
    _, _ = learner.learn(steps_of_no_change=100)
    option = np.argmax(learner.Q1 + learner.Q2, axis=1)
    option[goal] = -1
    return option


class PolicyIteration(object):
    def __init__(self, env, options=None, epsilon=0.1, gamma=0.90, alpha=0.1, surrogate_reward=None):
        self.epsilon = epsilon
        self.starting_epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.environment = env
        self.surrogate_reward = surrogate_reward
        self.num_states = env.observation_space.n
        self.available_actions = list(range(self.environment.action_space.n))
        if options is not None:
            self.available_actions.extend(options)

        self.V = np.zeros(self.num_states)
        self.pi = np.zeros(self.num_states, dtype=np.int)

    def _evalPolicy(self):
        delta = 0.0
        # for s in range(self.V.shape[0]):
        for s in range(self.num_states):
            old_v = self.V[s]
            action = self.available_actions[self.pi[s]]

            self.environment.force_state(s)
            nextState, nextReward, terminal, info = self.environment.step(action)
            if self.surrogate_reward is not None:
                nextReward = self.surrogate_reward(self.environment)

            new_v = nextReward + self.gamma * self.V[nextState]
            self.V[s] = new_v
            # if nextState == 4:
            #     print(self.V[s])
            delta = max(delta, abs(old_v - new_v))
        return delta

    def _improvePolicy(self):
        """ Policy improvement step. """
        policy_stable = True
        for s in range(self.num_states):
            old_action = self.pi[s]
            tempV = [0.0] * len(self.available_actions)
            # I first get all value-function estimates
            for i in range(len(self.available_actions)):

                self.environment.force_state(s)
                nextState, nextReward, terminal, info = self.environment.step(self.available_actions[i])
                if self.surrogate_reward is not None:
                    nextReward = self.surrogate_reward(self.environment)

                tempV[i] = nextReward + self.gamma * self.V[nextState]

            # Now I take the argmax
            self.pi[s] = np.argmax(tempV)
            # I break ties always choosing to terminate:
            if math.fabs(tempV[self.pi[s]] - tempV[(len(self.available_actions) - 1)]) < 0.001:
                self.pi[s] = (len(self.available_actions) - 1)
            if old_action != self.pi[s]:
                policy_stable = False

        return policy_stable

    def solvePolicyIteration(self, theta=0.001):
        """ Implementation of Policy Iteration, as in the policy iteration pseudo-code presented in Sutton and Barto
        (2016). """

        policy_stable = False
        history = collections.deque(maxlen=400)
        for _ in range(400): history.append(0)

        while not policy_stable:
            # Policy evaluation
            self.environment.render_board(some_matrix=self.V, policy=self.pi)
            delta = self._evalPolicy()
            # self.environment.print_board(some_matrix=self.V, policy=self.pi)
            it = 0
            while theta < delta or it > 1000:
                delta = self._evalPolicy()
                it += 1
                print(it, sum(history)/len(history))
                history.append(delta)
                if random.random() < 0.1:
                    self.environment.render_board(some_matrix=self.V, policy=self.pi)

            # Policy improvement
            policy_stable = self._improvePolicy()
            self.environment.render_board(some_matrix=self.V, policy=self.pi)
        return self.V, self.pi

    def learn(self, goal, steps_of_no_change=None):
        V, pi = self.solvePolicyIteration()

        # I'll assign the goal as the termination action
        pi[goal] = -1
        return pi
