import sys
import learners.double_q
import learners.policy_iter
import numpy as np
# from learning import Learning

import envs.gridworld


class Regressor(object):
    def __init__(self, env_distr):
        self.state_size = np.square(env_distr.invariants['size'])

    def get_reward_function(self, fitness):
        prediction = np.random.rand(self.state_size)

        # prediction = -np.ones(self.state_size)
        # prediction[4] = 1

        def reward_function(mdp):
            # return np.dot(mdp.state, prediction)
            return prediction[mdp.agent_position_idx]

        return reward_function


class GoalSelector(object):
    def __init__(self, num_goals):
        self.threshold = 1
        self.num_goals = num_goals

    def select_goals(self, mdp, reward_function):
        # pol_iter = learners.policyIteration(env=mdp, surrogate_reward=reward_function)
        learner = learners.double_q.DoubleQLearning(env=mdp, surrogate_reward=reward_function)
        goals = learner.learn(
            alpha=0.1,
            epsilon=0.1,
            gamma=1,  # 0.9,
            training_episodes_nr=100,
            num_goals=self.num_goals,
            goal_criterion=self.make_goal_criterion(self.threshold)
        )
        return goals

    @staticmethod
    def make_goal_criterion(threshold):
        def goal_criterion(state, delta_q):
            return delta_q > threshold

        return goal_criterion


class EnvGenerator(object):
    def __init__(self, klass, invariants):
        self.klass = klass
        self.invariants = invariants
        self.variables = {param[0]: param[1] for param in klass.get_params() if param[0] not in invariants.keys()}

    def gen_samples(self, training=True):
        while True:
            yield self.klass(**self.invariants, **self.variables)


def main():
    # env = envs.gridworld.GridWorld()
    mdp_distribution = EnvGenerator(envs.gridworld.GridWorld, invariants={'size': 12})
    regressor = Regressor(env_distr=mdp_distribution)
    goal_selector = GoalSelector(num_goals=3)
    TRAINING_SIZE = 5
    NR_EPOCHS = 5
    TEST_SIZE = 1

    fitness = None
    for epoch in range(NR_EPOCHS):
        reward_function = regressor.get_reward_function(fitness)
        for idx, mdp in enumerate(mdp_distribution.gen_samples(training=True)):
            if idx > TRAINING_SIZE:
                break
            goals = goal_selector.select_goals(mdp, reward_function)
            goals = [idx for score, idx in goals]
            print("GOALS:", goals)
            mdp.plot_goals(goals)
            options = []
            for goal in goals:
                print("generating policy for goal:", goal)

                def surrogate_reward(mdp):
                    return 1 if goal == mdp.s else -1

                skill = learners.policy_iter.policyIteration(env=mdp, surrogate_reward=surrogate_reward)
                options.append(skill)

            cum_cum_reward = 0
            for idx, mdp in enumerate(mdp_distribution.gen_samples(training=True)):
                if idx > TEST_SIZE:
                    break
                cum_cum_reward += learners.double_q.qLearningWithOptions(
                    env=mdp,
                    alpha=0.1,
                    gamma=0.9,
                    epsilon=0.1,
                    maxLengthEp=100,
                    nEpisodes=100,
                    loadedOptions=options
                )
            fitness = cum_cum_reward / TEST_SIZE
            print(fitness)

        # terminal = False
        # while not terminal:
        #     state, reward, terminal, info = env.step(1)
        #     env.render(mode="ansi")


if __name__ == "__main__":
    main()
