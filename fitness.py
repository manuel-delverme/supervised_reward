import matplotlib.pyplot as plt
import numpy as np

import config
import learners.approx_q_learning
import utils


def fitness_function(reward_vector):
    assert len(reward_vector.shape) == 1

    if reward_vector is None:
        intrinsic_reward_function = None
        generate_options = False
    else:
        reward_vector = np.array(reward_vector)
        generate_options = True

        def intrinsic_reward_function(observation, info):
            image = observation.reshape(-1)
            return reward_vector.dot(image)

        intrinsic_reward_function.reward_vector = reward_vector

    # draw training environment
    mdp = config.environment()

    if intrinsic_reward_function is None:
        options = []
    else:
        # generate options in that
        options, _, _, _ = learners.approx_q_learning.learn(
            environment=mdp, surrogate_reward=intrinsic_reward_function, training_steps=config.option_discovery_steps,
            generate_options=generate_options, eval_fitness=False)

    for idx, option in enumerate(options):
        motivating_vector = option.reward_matrix

        plt.suptitle(f'discovered option nr {idx}')
        for layer in range(motivating_vector.shape[2]):
            plt.subplot(2, 2, layer + 1)
            plt.imshow(motivating_vector[layer], vmin=motivating_vector.min(), vmax=motivating_vector.max())
            cbar = plt.colorbar()
        plt.show()
        utils.utils.enjoy_policy(mdp, option)

    env = config.environment()
    fitness = utils.utils.eval_options(env, options)
    print(fitness)
    return fitness, options
