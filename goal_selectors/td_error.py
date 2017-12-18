import learners


class TDErrorGoals(object):
    def __init__(self, num_goals):
        self.threshold = 1
        self.num_goals = num_goals

    def select_goals(self, mdp, reward_function):
        return [1, 31]
        # pol_iter = learners.policyIteration(env=mdp, surrogate_reward=reward_function)
        learner = learners.double_q.DoubleQLearning(env=mdp, surrogate_reward=reward_function)
        print("selecting goals")
        goals = learner.learn(
            alpha=0.1,
            epsilon=0.1,
            gamma=1,  # 0.9,
            training_episodes_nr=100,
            num_goals=self.num_goals,
            goal_criterion=self.make_goal_criterion(self.threshold)
        )
        goals = [idx for score, idx in goals]
        print("GOALS:", goals)
        mdp.plot_goals(goals)
        return goals

    @staticmethod
    def make_goal_criterion(threshold):
        def goal_criterion(state, delta_q):
            return delta_q > threshold

        return goal_criterion
