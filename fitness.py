import numpy as np

import config
import learners.approx_q_learning
import shared.utils

nr_fitness_calls = 0


def fitness_function(intrinsic_reward_function):
    global nr_fitness_calls
    # assert len(reward_vector.shape) == 1

    generate_options = True
    # draw training environment
    mdp = config.environment()

    if intrinsic_reward_function is None:
        raise Exception
        intrinsic_reward_function = None
        generate_options = False
        options = []
    else:
        # generate options in that
        nr_fitness_calls += 1
        options, _, _, _ = learners.approx_q_learning.learn(
            environment=mdp, surrogate_reward=intrinsic_reward_function, training_steps=config.option_discovery_steps,
            generate_options=generate_options, eval_fitness=False, log_postfix=f'generate_options_nr_{nr_fitness_calls}')

    # for idx, option in enumerate(options):
    #     motivating_vector = option.reward_matrix

    #     plt.suptitle(f'discovered option nr {idx}')
    #     for layer in range(motivating_vector.shape[2]):
    #         plt.subplot(2, 2, layer + 1)
    #         plt.imshow(motivating_vector[:, :, layer], vmin=motivating_vector.min(), vmax=motivating_vector.max())
    #         cbar = plt.colorbar()
    #     plt.show()
    #     shared.shared.enjoy_policy(mdp, option, reward_function=lambda x: x.ravel().dot(motivating_vector.reshape(-1)))

    env = config.environment()
    fitness = shared.utils.eval_options(env, options)
    # print(fitness)
    return fitness, options
