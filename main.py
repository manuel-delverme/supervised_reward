import functools
import itertools
from typing import Callable, List, Tuple
from utils import utils
import random
import numpy as np
import controller.meta_controller
import envs.boxes
import envs.minigrid
import envs.hungry_thirsty
import learners.q_learning
import config


# import multiprocessing.reduction
# multiprocessing.reduction.ForkingPickler


def main(experiment_id, population, training_steps, eval_training_steps, eval_test_steps, side_size,
         evolution_iters, env_name):
    possible_box_positions = list(
        itertools.combinations([0, side_size - 1, (side_size * side_size) - side_size, side_size * side_size - 1, ], 2)
    )

    if env_name == "hungry-thirsty":
        fitness_fn = generate_fitness_fn(
            functools.partial(envs.hungry_thirsty.HungryThirsty, side_size=side_size),
            eval_training_steps=eval_training_steps,
            eval_test_steps=eval_test_steps,
            training_steps=training_steps,
            possible_box_positions=possible_box_positions,
        )
        nr_tiles = side_size * side_size
        nr_agent_states = 4  # None, Hungry, Thristy, Hungry+Thristy
        reward_space_size = nr_tiles * nr_agent_states
    elif env_name == "boxes":
        fitness_fn = generate_fitness_fn(
            functools.partial(envs.boxes.BoxWorld, side_size=side_size),
            eval_training_steps=eval_training_steps,
            eval_test_steps=eval_test_steps,
            training_steps=training_steps,
            possible_box_positions=possible_box_positions,
        )
        nr_tiles = side_size * side_size
        nr_agent_states = 4  # None, Hungry, Thristy, Hungry+Thristy
        reward_space_size = nr_tiles * nr_agent_states
    elif env_name == "minigrid":
        fitness_fn = generate_fitness_fn(
            envs.minigrid.MiniGrid,
            eval_training_steps=eval_training_steps,
            eval_test_steps=eval_test_steps,
            training_steps=training_steps,
            possible_box_positions=[(None, None) for _ in range(6)],
        )
        nr_tiles = side_size * side_size
        nr_agent_states = 4  # None, Hungry, Thristy, Hungry+Thristy
        reward_space_size = nr_tiles * nr_agent_states
    elif env_name == "debug":
        def fitness_fn(reward_vector):
            if reward_vector is None:
                return 0, []
            else:
                return np.array(reward_vector).sum(), []
        reward_space_size = 100
    else:
        raise NotImplementedError("{} is not a valid environment".format(env_name))

    regressor = controller.meta_controller.GeneticEvolution(
        population_size=population,
        reward_space_size=reward_space_size,
    )
    regressor.optimize(experiment_id, fitness_function=fitness_fn, n_iterations=evolution_iters,)


def generate_fitness_fn(
        env_class,
        training_steps: int,
        eval_training_steps: int,
        eval_test_steps: int,
        possible_box_positions: List[Tuple[int, int]],
):
    _possible_box_positions = tuple(possible_box_positions)

    def fitness_fn(reward_vector):
        if reward_vector is None:
            intrinsic_reward_function = None
            generate_options = False
        else:
            reward_vector = np.array(reward_vector)
            generate_options = True

            def intrinsic_reward_function(state):
                return reward_vector[state]

        box_positions = list(_possible_box_positions)
        random.shuffle(box_positions)

        # to generator
        possible_box_positions = (p for p in box_positions)

        # draw training environment
        mdp = env_class(*next(possible_box_positions))

        # generate options in that
        learner = learners.q_learning.QLearning(env=mdp, surrogate_reward=intrinsic_reward_function)
        options, cum_reward, fitnesses = learner.learn(xs=[training_steps, ], generate_options=generate_options)

        fitness = utils.eval_options(env_class, options, possible_box_positions, xs=[eval_training_steps, ])
        return fitness, options

    return fitness_fn


if __name__ == "__main__":
    main(
        config.main.experiment_id,
        config.main.population,
        config.main.training_steps,
        config.main.eval_training_steps,
        config.main.eval_test_steps,
        config.main.side_size,
        config.main.evolution_iters,
        config.main.env_name
    )
