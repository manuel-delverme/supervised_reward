import numpy as np
import heapq
import tqdm
import random


class DoubleQLearning(object):
    def __init__(self, env, surrogate_reward=None):
        self.Q1 = 0.00001 * np.random.rand(env.observation_space.n, env.action_space.n)
        self.Q2 = 0.00001 * np.random.rand(env.observation_space.n, env.action_space.n)
        for state in env.terminal_states:
            self.Q1[state, :] = 0
            self.Q2[state, :] = 0
        self.environment = env
        self.surrogate_reward = surrogate_reward

    @staticmethod
    def null_criterion(*args, **kwargs):
        return False

    def learn(self, alpha, gamma, epsilon, training_episodes_nr, num_goals=10, goal_criterion=null_criterion):
        # returnSum = 0.0
        print("[double_q] learn")
        goals = []
        goal_states = set()
        for episode_num in tqdm.tqdm(range(training_episodes_nr)):
            old_state = self.environment.reset()
            terminal = False
            # if self.surrogate_reward is not None:
            #     reward = self.surrogate_reward(self.environment)

            while not terminal:
                if epsilon > random.random():
                    Q_total_s = self.Q1[old_state, :] + self.Q2[old_state, :]
                    action = np.argmax(Q_total_s)
                else:
                    action = random.randint(0, self.environment.action_space.n - 1)

                new_state, reward, terminal, info = self.environment.step(action)
                if self.surrogate_reward is not None:
                    reward = self.surrogate_reward(self.environment)

                if random.random() > 0.5:
                    Q, q = self.Q1, self.Q2
                else:
                    Q, q = self.Q2, self.Q1
                best_future_act = np.argmax(Q[new_state, :])
                old_Q = Q[old_state, action]
                delta_Q = alpha * (reward + (gamma * q[new_state, best_future_act]) - old_Q)
                old_Q += delta_Q

                # if goal_criterion(old_state, delta_Q):
                #     goals.append(old_state)
                # TODO: or new state?
                if len(goals) < num_goals and new_state not in goal_states:
                    goal_states.add(new_state)
                    heapq.heappush(goals, (delta_Q, new_state))
                elif delta_Q > goals[0][0]:
                    if new_state not in goal_states:
                        # print("replacing", goals[0], "with", (delta_Q, new_state))
                        goal_states.add(new_state)
                        removed_item = heapq.heapreplace(goals, (delta_Q, new_state))
                        goal_states.remove(removed_item[1])
                    else:
                        for goal_idx, (goal_dq, state_idx) in enumerate(goals):
                            if state_idx == new_state and goal_dq < delta_Q:
                                # print("updating", state_idx, "with", (delta_Q, new_state))
                                goals[goal_idx] = (delta_Q, state_idx)
                                heapq.heapify(goals)

                old_state = new_state
        return goals


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
    for episode_idx in tqdm.tqdm(range(n_episodes)):
        _ = learner.learn_one_episode()
    cum_reward = learner.evaluateOneEpisode(render=True)
    # learner.evaluateOneEpisode(render=True)
    return cum_reward


def is_skill(previous_action):
    return isinstance(previous_action, np.ndarray)


def is_terminal(skill, old_state):
    return skill[old_state] == -1


class QLearning:
    def __init__(self, alpha, gamma, epsilon, environment, options, time_limit):
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
        for state in environment.terminal_states:
            self.Q1[state, :] = 0
            self.Q2[state, :] = 0
        self.environment = environment

    def learn_one_episode(self):
        timestep = 0
        cumulativeReward = 0

        old_state = self.environment.reset()
        terminal = False
        while not terminal and timestep < self.time_limit:
            action, primitive_action = self.pick_action(old_state)

            new_state, reward, terminal, info = self.environment.step(primitive_action)
            cumulativeReward += reward

            if random.random() > 0.5:
                Q, q = self.Q1, self.Q2
            else:
                Q, q = self.Q2, self.Q1

            best_future_q = np.max(Q[new_state, :])
            old_Q = Q[old_state, action]
            delta_Q = self.alpha * (reward + (self.gamma * best_future_q) - old_Q)

            # update the value in self.Q1 or self.Q2 by pointer
            old_Q += delta_Q

            old_state = new_state
            timestep += 1
        cumulativeReward += (self.time_limit - timestep) * reward
        return cumulativeReward

    def pick_action(self, old_state):
        if is_skill(self.previous_action) and not is_terminal(self.previous_action, old_state):
            primitive_action = self.previous_action[old_state]
        else:
            action = None
            while action is not None and not is_terminal(action, old_state):  # TODO: refactor
                if self.epsilon > random.random():
                    Q_total_s = self.Q1[old_state, :] + self.Q2[old_state, :]
                    action = np.argmax(Q_total_s)
                else:
                    action = random.choice(self.action_list)

                if is_skill(action):
                    primitive_action = action[old_state]
                else:
                    primitive_action = action

            self.previous_action = action
        return primitive_action

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
