import learners.double_q_learning
import controller.meta_controller
import envs.hungry_thirsty
import envs.boxes
import numpy as np
import tqdm
import envs.gridworld
import time
import itertools


def main():
    TRAINING_SIZE = 1
    NR_EPOCHS = 1000
    TEST_SIZE = 11
    POPULATION_SIZE = 4

    # TEST RUN
    # mdp = envs.hungry_thirsty.HungryThirsty(side_size=6, water_position=0, food_position=1)
    mdp = envs.boxes.BoxWorld(side_size=6, box_positions=(0, 5))
    # learner = learners.double_q_learning.QLearning(env=mdp)
    # options, cum_reward = learner.learn(steps_of_no_change=100, generate_options=False)
    # print(cum_reward)
    # learner = learners.double_q_learning.QLearning(env=mdp)
    # options, cum_reward = learner.learn(steps_of_no_change=1000, max_steps=70000)


    def fitness_hungry_thirsty(reward_vector):
        mdp = envs.hungry_thirsty.HungryThirsty(side_size=6, water_position=0, food_position=5)

        def intrinsic_reward_function(_mdp):
            thirst = _mdp._state['thirsty']
            hunger = _mdp._state['hungry']
            x = np.array((
                thirst and hunger,
                not thirst and not hunger,
                thirst and not hunger,
                hunger and not thirst,
            ), dtype=np.int)
            # can be optimized as reward_vec[idx]
            return np.dot(reward_vector, x)

        learner = learners.double_q_learning.QLearning(env=mdp, surrogate_reward=intrinsic_reward_function)
        options, cum_reward = learner.learn(steps_of_no_change=100, max_steps=70000, generate_options=True)

        # eval options
        cum_cum_reward = 0
        possible_box_positions = itertools.combinations([0, 6, 30, 36], 2)
        for eval_step, box_positions in zip(range(TEST_SIZE), possible_box_positions):
            food_pos, water_pos = box_positions
            mdp = envs.hungry_thirsty.HungryThirsty(side_size=6, food_position=food_pos, water_position=water_pos)
            learner = learners.double_q_learning.QLearning(env=mdp, options=options)
            _, cum_reward = learner.learn(steps_of_no_change=100, generate_options=False)
            cum_cum_reward += cum_reward
        fitness = cum_cum_reward / TEST_SIZE
        # history.append(fitness)
        print("found {} options".format(len(options)), reward_vector, fitness)
        return fitness,

    def fitness_boxes(reward_vector):
        mdp = envs.boxes.BoxWorld(side_size=6, box_positions=(0, 5))

        def intrinsic_reward_function(_mdp):
            # thirst = _mdp._state['thirsty']
            hunger = _mdp._state['hungry']

            box1_pos, box2_pos =_mdp.box_positions
            box1 = _mdp._state['box'][box1_pos]
            box2 = _mdp._state['box'][box2_pos]
            world_states = []
            for _box1 in envs.boxes._BoxState:
                for _box2 in envs.boxes._BoxState:
                    for _hunger in (True, False):
                        world_states.append((box1 == _box1 and box2 == _box2 and hunger == _hunger))

            x = np.array(world_states, dtype=np.int)
            return np.dot(reward_vector, x)

        learner = learners.double_q_learning.QLearning(env=mdp, surrogate_reward=intrinsic_reward_function)
        options, cum_reward = learner.learn(steps_of_no_change=1000, max_steps=70000, generate_options=True)

        # eval options
        cum_cum_reward = 0
        possible_box_positions = itertools.combinations([0, 5, 30, 35], 2)
        for eval_step, box_positions in zip(range(TEST_SIZE), possible_box_positions):
            mdp = envs.boxes.BoxWorld(side_size=6, box_positions=box_positions)
            learner = learners.double_q_learning.QLearning(env=mdp, options=options)
            _, cum_reward = learner.learn(steps_of_no_change=100, generate_options=False)
            cum_cum_reward += cum_reward
        fitness = cum_cum_reward / TEST_SIZE
        # history.append(fitness)
        print("found {} options".format(len(options)), reward_vector, fitness)
        return fitness,

    regressor = controller.meta_controller.EvolutionaryAlgorithm(
        population_size=POPULATION_SIZE,
        fitness_function=fitness_boxes,
        reward_space_size=8,
    )
    regressor.optimize()

    # fitness = None
    # progress_bar = tqdm.tqdm()

    # terminal = False
    # while not terminal:
    #     state, reward, terminal, info = env.step(1)
    #     env.render(mode="ansi")


if __name__ == "__main__":
    main()
