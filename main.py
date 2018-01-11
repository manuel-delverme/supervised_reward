import itertools
import multiprocessing
import disk_utils
import pickle
import tqdm
import collections
import random
import numpy as np
import controller.meta_controller
import envs.boxes
import envs.gridworld
import envs.hungry_thirsty
import learners.double_q_learning
import numba


def main():
    POPULATION_SIZE = 4
    TRAINING_NO_CHANGE_STOP = 1000
    GENERATE_RANDOM_OPTIONS = True
    TRAINING_MAX_STEPS = 10000

    TEST_MAX_STEPS_TRAIN = 2000
    TEST_MAX_STEPS_EVAL = 1000
    SIDE_SIZE = 6

    env_name = "brute_boxworld"  # "hungry-thirsty"

    if env_name == "hungry-thirsty":
        def fitness_hungry_thirsty(reward_vector):
            # init a world
            possible_box_positions = [
                0,
                SIDE_SIZE - 1,
                SIDE_SIZE * SIDE_SIZE - 1,
                (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
            ]
            _box_positions = []
            for idx, box_pos in enumerate(possible_box_positions[:-1]):
                _box_positions.append((box_pos, possible_box_positions[idx + 1]))
                _box_positions.append((possible_box_positions[idx + 1], box_pos))

            random.shuffle(_box_positions)
            possible_box_positions = (p for p in _box_positions)

            water_pos, food_pos = next(possible_box_positions)

            if GENERATE_RANDOM_OPTIONS:
                options = pick_random_options()
            else:
                print("training with water: {} food {}".format(water_pos, food_pos))
                mdp = envs.hungry_thirsty.HungryThirsty(
                    side_size=SIDE_SIZE, water_position=water_pos, food_position=food_pos
                )

                # define an intrinsic reward fn
                def intrinsic_reward_function(_mdp):
                    thirst = _mdp._state['thirsty']
                    hunger = _mdp._state['hungry']
                    x = np.array((
                        thirst and hunger,
                        not thirst and not hunger,
                        thirst and not hunger,
                        hunger and not thirst,
                    ), dtype=np.int)
                    # TODO: should be optimized as reward_vec[idx]
                    return np.dot(reward_vector, x)

                # generate option set
                learner = learners.double_q_learning.QLearning(
                    env=mdp,
                    surrogate_reward=intrinsic_reward_function,
                    train_run=True,
                )
                options, cum_reward = learner.learn(
                    steps_of_no_change=TRAINING_NO_CHANGE_STOP,
                    max_steps=TRAINING_MAX_STEPS,
                    generate_options=True
                )

            # eval options

            # cum_cum_reward += cum_reward
            # num_of_test_samples += 1

            cum_cum_reward = 0
            print_statistics(-1, options)
            for eval_step, box_positions in enumerate(possible_box_positions):
                food_pos, water_pos = box_positions

                mdp = envs.hungry_thirsty.HungryThirsty(side_size=6, food_position=food_pos, water_position=water_pos)
                learner = learners.double_q_learning.QLearning(env=mdp, options=options, test_run=True)
                _, _ = learner.learn(max_steps=TEST_MAX_STEPS_TRAIN)
                cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
                cum_cum_reward += cum_reward

            fitness = cum_cum_reward / (eval_step + 1)
            print_statistics(fitness, options)
            return fitness,

        fitness_fn = fitness_hungry_thirsty
        reward_space_size = 4

    elif env_name == "boxworld":
        def fitness_boxes(reward_vector):
            # init a world
            possible_box_positions = list(itertools.combinations([
                0,
                SIDE_SIZE - 1,
                (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                SIDE_SIZE * SIDE_SIZE - 1,
            ], 2))
            random.shuffle(possible_box_positions)
            possible_box_positions = (p for p in possible_box_positions)

            training_sample = next(possible_box_positions)

            if GENERATE_RANDOM_OPTIONS:
                options = pick_random_options()
            else:
                mdp = envs.boxes.BoxWorld(side_size=6, box_positions=training_sample)

                # define reward fn
                def intrinsic_reward_function(_mdp):
                    # thirst = _mdp._state['thirsty']
                    hunger = _mdp._state['hungry']

                    box1_pos, box2_pos = _mdp.box_positions
                    box1 = _mdp._state['box'][box1_pos]
                    box2 = _mdp._state['box'][box2_pos]
                    # world_states = []
                    _hack_idx = 0
                    for _box1 in envs.boxes._BoxState:
                        for _box2 in envs.boxes._BoxState:
                            for _hunger in (True, False):
                                # world_states.append((box1 == _box1 and box2 == _box2 and hunger == _hunger))
                                if box1 == _box1 and box2 == _box2 and hunger == _hunger:
                                    _idx = _hack_idx
                                _hack_idx += 1

                    # x = np.array(world_states, dtype=np.int)
                    # return np.dot(reward_vector, x)
                    return reward_vector[_idx]

                # generate options set
                learner = learners.double_q_learning.QLearning(env=mdp, surrogate_reward=intrinsic_reward_function,
                                                               train_run=True)
                options, cum_reward = learner.learn(steps_of_no_change=1000, max_steps=10000, generate_options=True)

            # eval options
            cum_cum_reward = 0
            for eval_step, box_positions in enumerate(possible_box_positions):
                mdp = envs.boxes.BoxWorld(side_size=6, box_positions=box_positions)
                learner = learners.double_q_learning.QLearning(env=mdp, options=options, test_run=True)
                _, _ = learner.learn(max_steps=TEST_MAX_STEPS_TRAIN, generate_options=False, plot_progress=False)

                cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
                cum_cum_reward += cum_reward
            fitness = cum_cum_reward / eval_step

            print_statistics(fitness, options)
            return fitness

        fitness_fn = fitness_boxes
        reward_space_size = 18
    else:
        raise NotImplementedError("{} is not a valid environment".format(env_name))

    regressor = controller.meta_controller.CMAES(
        population_size=POPULATION_SIZE,
        fitness_function=fitness_fn,
        reward_space_size=reward_space_size,
    )
    regressor.optimize()


def bruteforce_options():
    number_of_options = 3
    TEST_MAX_STEPS_EVAL = 100
    SIDE_SIZE = 6
    scores = collections.defaultdict(dict)

    option_sets = itertools.combinations([None] * (number_of_options) + list(range(36)), number_of_options)
    option_sets = list(option_sets)
    possible_box_positions = list(itertools.combinations([0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                                                          SIDE_SIZE * SIDE_SIZE - 1, ], 2))

    xs = (10, 100, 200, 300, 1000, 10000)
    progress = tqdm.tqdm(total=len(option_sets) * len(xs))

    mdp = envs.boxes.BoxWorld(side_size=6, box_positions=())
    learner = learners.double_q_learning.QLearning(env=mdp, options=[], test_run=True)

    option_map = {
        tuple(): tuple()
    }

    for o in range(36):
        mdp.agent_position_idx = o
        learner.generate_option()
        option_vec = tuple(learner.available_actions[-1])
        option_map[o] = option_vec

    option_sets = [tuple(o) for o in option_sets]
    option_sets = [tuple(o for o in option_set if o is not None) for option_set in option_sets]
    option_vecs = [tuple(option_map[o] for o in option_set) for option_set in option_sets]

    for option_ids, option_vec in zip(option_sets, option_vecs):
        cum_scores = collections.defaultdict(float)
        for eval_step, box_positions in enumerate(possible_box_positions):
            option_set_score = eval_option_on_mdp(TEST_MAX_STEPS_EVAL, box_positions, option_vec, xs)
            # fitness = cum_cum_reward / eval_step
            for k in option_set_score.keys():
                cum_scores[k] += option_set_score[k]
            progress.update(1)
        scores[option_ids] = cum_scores
        # print_statistics(fitness, option_set)
    return scores


@disk_utils.disk_cache
@numba.jit
def eval_option_on_mdp(TEST_MAX_STEPS_EVAL, box_positions, option_vec, xs):
    option_set_score = {}
    mdp = envs.boxes.BoxWorld(side_size=6, box_positions=box_positions)
    learner = learners.double_q_learning.QLearning(env=mdp, options=option_vec, test_run=True)

    for test_max_steps_train in xs:
        learner.learn(max_steps=test_max_steps_train)
        cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
        option_set_score[test_max_steps_train] = cum_reward
        # cum_cum_reward += cum_reward
    return option_set_score


def print_statistics(fitness, options):
    option_names = []
    for option in options:
        option_names.append(int(np.argwhere(option == -1)[0]))
    option_names = " ".join(str(n) for n in sorted(option_names))
    print("score:\t{}\toptions: {}\t{}".format(fitness, len(options), option_names))


def pick_random_options():
    mdp = envs.boxes.BoxWorld(side_size=6, box_positions=())
    options = []
    # for goal in random.sample(range(mdp.number_of_tiles), random.randrange(1, 4)):
    for goal in (0, 5, 30, 35):
        opt = learners.double_q_learning.learn_option(goal, mdp)
        # TODO: REMOVE HACK
        if opt.shape[0] < mdp.observation_space.n:
            # TODO: remove print("OPTION SIZE MISMATCH, TILING")
            opt = np.tile(
                opt[:mdp.number_of_tiles],
                mdp.observation_space.n // mdp.number_of_tiles
            )
        options.append(opt)
    return options


if __name__ == "__main__":
    # main()
    bruteforce_options()
