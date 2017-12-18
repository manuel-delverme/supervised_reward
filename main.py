import sys
import disk_utils
import collections
import learners.double_q
import learners.policy_iter
import controller.meta_controller
import goal_selectors.td_error
import mdp_generator.env_generator
import numpy as np
# from learning import Learning

import envs.gridworld


def main():
    TRAINING_SIZE = 1
    NR_EPOCHS = 1000
    TEST_SIZE = 1
    NUM_GOALS = 2
    POPULATION_SIZE = 4

    history = collections.deque(maxlen=100)

    # env = envs.gridworld.GridWorld()
    mdp_distribution = mdp_generator.env_generator.EnvGenerator(envs.gridworld.GridWorld, invariants={'size': 12})
    regressor = controller.meta_controller.EvolutionaryAlgorithm(env_distr=mdp_distribution, population_size=POPULATION_SIZE)
    goal_selector = goal_selectors.td_error.TDErrorGoals(num_goals=NUM_GOALS)

    fitness = None
    reward_function_gen = regressor.get_reward_function()
    for epoch in range(NR_EPOCHS):
        for eval_step, mdp in enumerate(mdp_distribution.gen_samples(training=True)):
            reward_function = reward_function_gen.send(fitness)
            if eval_step > TRAINING_SIZE:
                break
            # goals = select_goals(goal_selector, mdp, reward_function)
            goals = goal_selector.select_goals(mdp, reward_function)

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
