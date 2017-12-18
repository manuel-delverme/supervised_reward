import sys
import disk_utils
import collections
import learners.double_q
import learners.policy_iter
import controller.meta_controller
import numpy as np
# from learning import Learning

import envs.gridworld

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
    TRAINING_SIZE = 1
    NR_EPOCHS = 1000
    TEST_SIZE = 1
    NUM_GOALS = 2
    POPULATION_SIZE = 4
    history = collections.deque(maxlen=100)

    # env = envs.gridworld.GridWorld()
    mdp_distribution = EnvGenerator(envs.gridworld.GridWorld, invariants={'size': 12})
    regressor = controller.meta_controller.EvolutionaryAlgorithm(
        env_distr=mdp_distribution,
        population_size=POPULATION_SIZE
    )
    # regressor.stuff()
    goal_selector = GoalSelector(num_goals=NUM_GOALS)

    fitness = None
    reward_function_gen = regressor.get_reward_function()
    for epoch in range(NR_EPOCHS):
        for eval_step, mdp in enumerate(mdp_distribution.gen_samples(training=True)):
            reward_function = reward_function_gen.send(fitness)
            if eval_step > TRAINING_SIZE:
                break
            goals = select_goals(goal_selector, mdp, reward_function)

            options = learn_skills(goals, mdp)

            print("testing goals")
            cum_cum_reward = 0
            for eval_step, mdp in enumerate(mdp_distribution.gen_samples(training=True)):
                if eval_step >= TEST_SIZE:
                    break
                cum_cum_reward += learners.double_q.q_learning_with_options(
                    env=mdp,
                    alpha=0.5,
                    gamma=0.99,
                    epsilon=0.1,
                    time_limit=100,
                    n_episodes=100,
                    options=options
                )
            fitness = cum_cum_reward / TEST_SIZE
            history.append(fitness)
            print(history)

        # terminal = False
        # while not terminal:
        #     state, reward, terminal, info = env.step(1)
        #     env.render(mode="ansi")


def select_goals(goal_selector, mdp, reward_function):
    return [1, 31]
    print("selecting goals")
    goals = goal_selector.select_goals(mdp, reward_function)
    goals = [idx for score, idx in goals]
    print("GOALS:", goals)
    mdp.plot_goals(goals)
    return goals


@disk_utils.disk_cache
def learn_skills(goals, mdp):
    print("learning goals")
    options = []
    for goal in goals:
        print("generating policy for goal:", goal)

        def surrogate_reward(mdp):
            return 1 if goal == mdp.s else -1

        skill = learners.policy_iter.policyIteration(env=mdp, surrogate_reward=surrogate_reward)
        options.append(skill)
    return options


if __name__ == "__main__":
    main()
