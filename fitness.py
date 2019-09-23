import config
import learners.approx_q_learning
import shared.utils


def fitness_function(intrinsic_reward_function):
    if intrinsic_reward_function.fitness is None or config.recalculate_fitness:
        mdp = config.environment()

        # generate options in that
        options, _, _, _ = learners.approx_q_learning.learn(
            environment=mdp, surrogate_reward=intrinsic_reward_function, training_steps=config.option_discovery_steps,
            generate_options=True, eval_fitness=False, log_postfix=f'generate_options_nr_{intrinsic_reward_function.reward_coords}')

        env = config.environment()
        intrinsic_reward_function.options = options
        if not options:
            intrinsic_reward_function.fitness = -shared.utils.eval_options(env, options)
        else:
            intrinsic_reward_function.fitness = shared.utils.eval_options(env, options)

    print(intrinsic_reward_function.fitness, intrinsic_reward_function.options)
    return intrinsic_reward_function.fitness, intrinsic_reward_function.options
