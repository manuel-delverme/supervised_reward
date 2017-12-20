import sys
import collections
import learners.double_q
import controller.meta_controller
import goal_selectors.td_error
import mdp_generator.env_generator
import numpy as np
import tqdm
# from learning import Learning

import envs.gridworld


def main():
    TRAINING_SIZE = 1
    NR_EPOCHS = 1000
    TEST_SIZE = 1
    NUM_GOALS = 2
    POPULATION_SIZE = 10

    history = collections.deque(maxlen=15)

    # env = envs.gridworld.GridWorld()
    mdp_distribution = mdp_generator.env_generator.EnvGenerator(envs.gridworld.GridWorld, invariants={'size': 6})
    option_generator = goal_selectors.td_error.TDErrorGoals(num_goals=NUM_GOALS)

    def eval_reward_function(reward_vector):
        mdp = next(mdp_distribution.gen_samples(training=True))

        def intrinsic_reward_function(mdp):
            thirst = mdp._state['thirsty']
            hunger = mdp._state['hungry']
            x = np.array((
                thirst and hunger,
                not thirst and not hunger,
                thirst and not hunger,
                hunger and not thirst,
            ), dtype=np.int)
            # can be optimized as reward_vec[idx]
            return np.dot(reward_vector, x)

        # turn reward into options
        options = option_generator.generate_options(mdp, intrinsic_reward_function, training_steps=1000)
        # progress_bar.set_description("found: {} options\n history: {}\n pop:\n{}\n".format(
        #     len(options),
        #     history,
        #     "\n".join([str(el[0]) for el in regressor.population])
        # ))

        # eval options
        cum_cum_reward = 0
        for eval_step, mdp in zip(range(TEST_SIZE), mdp_distribution.gen_samples(training=False)):
            learner = learners.double_q.DoubleQLearning(env=mdp, options=options)
            _, cum_reward = learner.learn(training_steps=1000)
            cum_cum_reward += cum_reward
        fitness = cum_cum_reward / TEST_SIZE
        # history.append(fitness)
        print("found {} options".format(len(options)), reward_vector, fitness)
        return fitness,

    regressor = controller.meta_controller.EvolutionaryAlgorithm(
        population_size=POPULATION_SIZE,
        fitness_function=eval_reward_function,
    )
    regressor.optimize()

    fitness = None
    progress_bar = tqdm.tqdm()

    # terminal = False
    # while not terminal:
    #     state, reward, terminal, info = env.step(1)
    #     env.render(mode="ansi")


if __name__ == "__main__":
    main()
