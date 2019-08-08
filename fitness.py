import config
import learners.approx_q_learning
import shared.utils

nr_fitness_calls = 0


def fitness_function(intrinsic_reward_function):
    global nr_fitness_calls

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
            generate_options=True, eval_fitness=False, log_postfix=f'generate_options_nr_{nr_fitness_calls}')

    env = config.environment()
    fitness = shared.utils.eval_options(env, options)
    return fitness, options
