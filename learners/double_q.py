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

    def learn(self, alpha, gamma, epsilon, training_episodes_nr, goal_criterion=null_criterion):
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
                    qTotal = self.Q1 + self.Q2
                    action = np.argmax(qTotal[old_state, :])
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
                if len(goals) < 10 and new_state not in goal_states:
                    goal_states.add(new_state)
                    heapq.heappush(goals, (delta_Q, new_state))
                elif delta_Q > goals[0][0]:
                    if new_state not in goal_states:
                        print("replacing", goals[0], "with", (delta_Q, new_state))
                        goal_states.add(new_state)
                        removed_item = heapq.heapreplace(goals, (delta_Q, new_state))
                        goal_states.remove(removed_item[1])
                    else:
                        for goal_idx, (goal_dq, state_idx) in enumerate(goals):
                            if state_idx == new_state and goal_dq < delta_Q:
                                print("updating", state_idx, "with", (delta_Q, new_state))
                                goals[goal_idx] = (delta_Q, state_idx)
                                heapq.heapify(goals)

                old_state = new_state
        return goals
