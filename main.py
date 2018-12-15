import itertools
from utils import utils
import random
import numpy as np
import controller.meta_controller
import envs.boxes
import envs.gridworld
import envs.hungry_thirsty
import learners.q_learning
import gin


@gin.configurable
def main(experiment_id, cmaes_population, training_steps, eval_training_steps, eval_test_steps, side_size,
         evolution_iters, env_name):
    if env_name == "hungry-thirsty":
        fitness_fn = generate_fitness_fn(
            envs.hungry_thirsty.HungryThirsty,
            eval_training_steps=eval_training_steps,
            eval_test_steps=eval_test_steps,
            training_steps=training_steps,
        )

        nr_tiles = side_size * side_size
        nr_boxes = 2
        nr_box_states = 3
        reward_space_size = nr_tiles * nr_boxes * nr_box_states

    elif env_name == "boxworld":
        fitness_fn = generate_boxworld_fitness_fn()
        reward_space_size = side_size
    else:
        raise NotImplementedError("{} is not a valid environment".format(env_name))

    possible_box_positions = list(
        itertools.combinations([0, side_size - 1, (side_size * side_size) - side_size, side_size * side_size - 1, ], 2))

    regressor = controller.meta_controller.CMAES(
        population_size=cmaes_population,
        reward_space_size=reward_space_size,
    )
    regressor.optimize(
        experiment_id,
        fitness_function=fitness_fn,
        n_iterations=evolution_iters,
        mdp_parameters={
            'side_size': side_size,
            'possible_box_positions': tuple(possible_box_positions),
        },
    )


def generate_boxworld_fitness_fn():
    def fitness_boxes(args):
        args_dict = {k: v for k, v in args}
        del args

        reward_vector = np.array(args_dict['weights'])
        possible_box_positions = list(args_dict['possible_box_positions'])

        side_size = args_dict['side_size']
        random.shuffle(possible_box_positions)
        possible_box_positions = (p for p in possible_box_positions)

        training_sample = next(possible_box_positions)
        mdp = envs.boxes.BoxWorld(side_size=side_size, box_positions=training_sample)

        def intrinsic_reward_function(state):
            return reward_vector[state]

        # generate options set
        learner = learners.double_q_learning.QLearning(env=mdp, surrogate_reward=intrinsic_reward_function,
                                                       train_run=True)

        options, cum_reward = learner.learn(max_steps=10000, generate_options=True)

        # eval options
        cum_cum_reward = 0
        for eval_step, box_positions in enumerate(possible_box_positions):
            mdp = envs.boxes.BoxWorld(side_size=side_size, box_positions=box_positions)
            learner = learners.double_q_learning.QLearning(env=mdp, options=options, test_run=True)
            _, _ = learner.learn(max_steps=eval_training_steps, generate_options=False, plot_progress=False)

            cum_reward = learner.test(eval_steps=eval_test_steps)
            cum_cum_reward += cum_reward
        fitness = cum_cum_reward / eval_step

        print_statistics(fitness, options)
        return fitness

    return fitness_boxes


def generate_fitness_fn(
        env_class,
        training_steps: int,
        eval_training_steps: int,
        eval_test_steps: int,
):
    def fitness_hungry_thirsty(args):
        args_dict = {k: v for k, v in args}
        del args

        reward_vector = args_dict['weights']
        possible_box_positions = list(args_dict['possible_box_positions'])

        side_size = args_dict['side_size']
        random.shuffle(possible_box_positions)
        possible_box_positions = (p for p in possible_box_positions)

        box1, box2 = next(possible_box_positions)
        mdp = env_class(
            side_size=side_size, box1=box1, box2=box2
        )

        if reward_vector is None:
            intrinsic_reward_function = None
        else:
            reward_vector = np.array(reward_vector)

            def intrinsic_reward_function(state):
                return reward_vector[state]

        learner = learners.q_learning.QLearning(env=mdp, surrogate_reward=intrinsic_reward_function)
        options, cum_reward, fitnesses = learner.learn(xs=[training_steps, ], generate_options=True)

        def _load_env(params):
            food_pos, water_pos = params
            mdp = envs.hungry_thirsty.HungryThirsty(side_size=side_size, box2=food_pos, box1=water_pos)
            return mdp

        fitness = utils.eval_options(_load_env, options, possible_box_positions, xs=[eval_training_steps, ])
        return fitness

    return fitness_hungry_thirsty


if __name__ == "__main__":
    gin.parse_config_file('config.gin')
    main()
