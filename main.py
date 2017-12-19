import sys
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

    history = collections.deque(maxlen=15)

    # env = envs.gridworld.GridWorld()
    mdp_distribution = mdp_generator.env_generator.EnvGenerator(envs.gridworld.GridWorld, invariants={'size': 6})
    regressor = controller.meta_controller.EvolutionaryAlgorithm(env_distr=mdp_distribution, population_size=POPULATION_SIZE)
    option_generator = goal_selectors.td_error.TDErrorGoals(num_goals=NUM_GOALS)

    fitness = None
    reward_function_gen = regressor.get_reward_function()
    for epoch in range(NR_EPOCHS):
        for eval_step, mdp in enumerate(mdp_distribution.gen_samples(training=True)):
            intrinsic_reward_function = reward_function_gen.send(fitness)
            if eval_step > TRAINING_SIZE:
                break
            options = option_generator.generate_options(
                mdp,
                intrinsic_reward_function,
                training_steps=1000
            )
            print("found {} options".format(len(options)))
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



if __name__ == "__main__":
    main()
