import itertools
import numpy as np
import controller.meta_controller
import envs.gridworld
import envs.hungry_thirsty
import envs.simple_boxes
import learners.double_q_learning
import learners.q_learning


def evolve_intrinsic():
    POPULATION_SIZE = 8  # if < 6 cmaes mirrors, disable that (?)
    TRAINING_NO_CHANGE_STOP = 1000
    GENERATE_RANDOM_OPTIONS = False
    TRAINING_MAX_STEPS = 10000

    TEST_MAX_STEPS_TRAIN = 2000
    TEST_MAX_STEPS_EVAL = 1000
    OPTION_LEARNING_STEPS = 10000
    SIDE_SIZE = 6

    # fitness_fn, reward_space_size = envs.hungry_thirsty.get_env_fitness_fn(SIDE_SIZE)
    # fitness_fn, reward_space_size = envs.boxes.BoxWorld.get_fitness_fn(SIDE_SIZE), 18

    fitness_fn, reward_space_size = envs.simple_boxes.BoxWorldSimple.get_fitness_fn(SIDE_SIZE), 9

    regressor = controller.meta_controller.CMAES(
        population_size=POPULATION_SIZE,
        fitness_function=fitness_fn,
        reward_space_size=reward_space_size,
    )
    regressor.optimize()


def print_statistics(fitness, options):
    option_names = []
    for option in options:
        for idx, action in enumerate(option):
            if action == -1:
                option_names.append(idx)
                break
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


def test_qlearning():
    SIDE_SIZE = 6
    TEST_MAX_STEPS_TRAIN = 2000
    TEST_MAX_STEPS_EVAL = 1000

    possible_box_positions = list(itertools.combinations(
        [0, SIDE_SIZE - 1, (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE, SIDE_SIZE * SIDE_SIZE - 1, ], 2))
    cum_cum_reward = 0

    token_mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=(1, 2))
    learner = learners.double_q_learning.QLearning(env=token_mdp, options=[], test_run=True)
    token_mdp.agent_position_idx = 0
    learner.generate_option()
    option_vec0 = tuple(learner.available_actions[-1])
    token_mdp.agent_position_idx = 17
    learner.generate_option()
    option_vec1 = tuple(learner.available_actions[-1])
    option_vec = [option_vec0, option_vec1]

    for eval_step, box_positions in enumerate(possible_box_positions):
        mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=box_positions)
        learner = learners.q_learning.QLearning(env=mdp, options=option_vec, test_run=True)
        learner.learn(max_steps=TEST_MAX_STEPS_TRAIN)

        cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
        cum_cum_reward += cum_reward
    fitness_q = cum_cum_reward / eval_step
    print(fitness_q)

    cum_cum_reward = 0
    for eval_step, box_positions in enumerate(possible_box_positions):
        mdp = envs.simple_boxes.BoxWorldSimple(side_size=6, box_positions=box_positions)
        learner = learners.double_q_learning.QLearning(env=mdp, options=option_vec, test_run=True)
        learner.learn(max_steps=TEST_MAX_STEPS_TRAIN, generate_options=False)

        cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
        cum_cum_reward += cum_reward
    fitness_2q = cum_cum_reward / eval_step
    print("q, 2q", fitness_q, fitness_2q)


if __name__ == "__main__":
    print("rararara")
    # evolve_intrinsic()
    # update the genetic search, and plot where the masks activate
    # test_qlearning()
