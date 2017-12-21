import learners
import time


class TDErrorGoals(object):
    def __init__(self, num_goals):
        self.threshold = 1
        self.num_goals = num_goals

    def generate_options(self, mdp, reward_function, training_steps):
        # return [1, 31]
        # pol_iter = learners.policyIteration(env=mdp, surrogate_reward=reward_function)
        learner = learners.double_q.DoubleQLearning(env=mdp, surrogate_reward=reward_function)
        # print("selecting goals")
        options, cum_reward = learner.learn(
            steps_of_no_change=100,
            goal_criterion=self.make_goal_criterion(self.threshold)
        )
        # TODO: plot learned options
        for option in options:
            mdp.plot_policy(option)
            time.sleep(1)
        return options

    @staticmethod
    def make_goal_criterion(threshold):
        def goal_criterion(state, delta_q):
            return delta_q > threshold

        return goal_criterion
