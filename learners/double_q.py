import numpy as np
import disk_utils
import heapq
import random
import learners.policy_iter


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
        self.available_actions = list(range(0, self.environment.action_space.n - 1))
        if options is not None:
            self.available_actions.extend(options)

    @staticmethod
    def null_criterion(*args, **kwargs):
        return False

    def pick_action(self, old_state):
        if is_skill(self.previous_action) and not is_terminate_option(self.previous_action, old_state):
            primitive_action = self.previous_action[old_state]
        else:
            if self.epsilon > random.random():

                Q_total_s = self.Q1[old_state, :] + self.Q2[old_state, :]
                action = np.argmax(Q_total_s)
            else:
                action = random.choice(self.available_actions)

            if is_skill(action):
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

            if is_skill(action):
                time_steps_under_option += 1
            else:
                if random.random() > 0.5:
                    Q, q = self.Q1, self.Q2
                else:
                    Q, q = self.Q2, self.Q1

                best_future_q = np.max(Q[new_state, :])
                old_Q = Q[old_state, action]
                k = 1 + time_steps_under_option
                delta_Q = reward + ((self.gamma ** k) * best_future_q) - old_Q

                # update the value in self.Q1 or self.Q2 by pointer
                old_Q += self.alpha * delta_Q

                # TODO: or new state?
                if goal_criterion is not None and goal_criterion(old_state, delta_Q):
                    discovered_options.append(learn_option(old_state, self.environment))
                old_state = new_state
                # cumulative_reward += (self.time_limit - timestep) * reward
        return discovered_options, cumulative_reward


def q_learning_with_options(env, alpha, gamma, epsilon, n_episodes, time_limit, options):
    learner = QLearning(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        environment=env,
        time_limit=time_limit,
        options=options,
    )
    # learner = DoubleQLearning(environment=env, options)

    # cum_reward = 0
    # returns_learn = np.zeros(len(options), n_episodes)
    for episode_idx in range(n_episodes):
        _ = learner.learn_one_episode()
    cum_reward = learner.evaluateOneEpisode(render=True)
    # learner.evaluateOneEpisode(render=True)
    return cum_reward


def is_skill(previous_action):
    return isinstance(previous_action, np.ndarray)


def is_terminate_option(skill, old_state):
    return skill[old_state] == -1


class QLearning:
    def __init__(self, alpha, gamma, epsilon, environment, options, time_limit):
        raise NotImplementedError()
        self.environment = environment
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.options = options
        self.time_limit = time_limit
        self.previous_action = None
        num_states = self.environment.observation_space.n
        self.num_primitive_actions = self.environment.action_space.n
        self.Q1 = 0.00001 * np.random.rand(num_states, self.num_primitive_actions + len(options))
        self.Q2 = 0.00001 * np.random.rand(num_states, self.num_primitive_actions + len(options))
        self.action_list = list(range(self.num_primitive_actions)) + options
        if hasattr(environment, 'terminal_states'):
            for state in environment.terminal_states:
                self.Q1[state, :] = 0
                self.Q2[state, :] = 0
        self.environment = environment

    def learn_one_episode(self):
        timestep = 0
        cumulativeReward = 0

        old_state = self.environment.reset()
        terminal = False
        timesteps_under_option = 0
        while not terminal and timestep < self.time_limit:
            action, primitive_action = self.pick_action(old_state)

            new_state, reward, terminal, info = self.environment.step(primitive_action)
            cumulativeReward += reward

            if is_skill(action):
                timesteps_under_option += 1
            else:
                if random.random() > 0.5:
                    Q, q = self.Q1, self.Q2
                else:
                    Q, q = self.Q2, self.Q1

                best_future_q = np.max(Q[new_state, :])
                old_Q = Q[old_state, action]
                delta_Q = self.alpha * (reward + ((self.gamma ** timesteps_under_option) * best_future_q) - old_Q)
                # delta_Q = self.alpha * (reward + (self.gamma * best_future_q) - old_Q)

                # update the value in self.Q1 or self.Q2 by pointer
                old_Q += delta_Q

            old_state = new_state
            timestep += 1
        cumulativeReward += (self.time_limit - timestep) * reward
        return cumulativeReward

    def evaluateOneEpisode(self, eps=None, render=False):
        """Evaluate Q-learning for one episode."""

        reward = 0
        timestep = 0
        previous_action = -1
        cumulativeReward = 0

        old_state = self.environment.reset()
        terminal = False

        while not terminal and timestep < self.time_limit:
            if previous_action < self.numPrimitiveActions:
                action = self.epsilon_greedy(self.Q[old_state], old_state, epsilon=eps)
            action = self.get_primitive_action(old_state, action)

            if action == 'terminate':
                action = self.epsilon_greedy(self.Q[old_state], old_state, epsilon=eps)
                if isinstance(action, np.ndarray):
                    action = action[old_state]
                return action

            previous_action = action
            new_state, reward, terminal, info = self.environment.step(action)
            if render:
                self.environment.render()
            # reward = self.environment.step(action)
            cumulativeReward += reward
            # sNext = self.environment.getCurrentState()

            old_state = new_state
            timestep += 1
        return cumulativeReward


@disk_utils.disk_cache
def learn_option(goal, mdp):
    # print("generating policy for goal:", goal)

    def surrogate_reward(_mdp):
        return 1 if goal == _mdp.agent_position_idx else -1

    option = learners.policy_iter.policyIteration(env=mdp, surrogate_reward=surrogate_reward)
    return option
