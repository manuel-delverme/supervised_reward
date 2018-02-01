import enum
import numpy as np
import itertools
import envs.gridworld
import learners
import random
import gym.spaces


class BoxWorldActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class _BoxState(enum.Enum):
    OPEN = 0
    CLOSED = 2


class BoxWorldSimple(envs.gridworld.GridWorld):
    def __init__(self, side_size, box_positions=(0, 30), agent_lifetime=15):
        super(BoxWorldSimple, self).__init__(
            side_size=side_size,
            terminal_states=(),
            base_transition_probability=0.9,
        )
        self._lifetime = agent_lifetime
        self.time_left = agent_lifetime
        self.box1_is_full = None
        self.box2_is_full = None
        self.box_positions = box_positions

        self.reset()

        extra_dof = (len(_BoxState) ** 2)
        self.observation_space = gym.spaces.Discrete(self.observation_space.n * extra_dof)
        self.action_space = gym.spaces.Discrete(len(BoxWorldActions))

    def _step(self, action):
        assert self.time_left >= 0

        tile_idx, gridworld_reward, gridworld_terminal, info = super(envs.gridworld.GridWorld, self)._step(action)

        reward = -1
        if self.box1_is_full and self.box_positions[0] == self.agent_position_idx:
            self.box1_is_full = False
            reward = 100
        elif self.box2_is_full and self.box_positions[1] == self.agent_position_idx:
            self.box2_is_full = False
            reward = 100

        # open boxes randomly close
        if random.random() < 0.1:
            self.box1_is_full = True

        if random.random() < 0.1:
            self.box2_is_full = True

        state_obj = self._hash_state()

        if reward > 0:
            self.time_left -= 1
        if self.time_left < 0:
            terminal = True
        else:
            terminal = False

        return state_obj, reward, terminal, info

    def _hash_state(self):
        offset = 1
        state_hash = self.agent_position_idx * offset
        offset *= self.number_of_tiles

        state_hash += self.box1_is_full * offset
        offset *= 2

        state_hash += self.box2_is_full * offset
        # offset *= 2

        return state_hash

    def reset(self):
        super(BoxWorldSimple, self).reset()
        self.time_left = self._lifetime
        self.box1_is_full = True
        self.box2_is_full = True
        return self._hash_state()

    def show_board(self, some_matrix=None, close=False, policy=None, highlight_square=None, info={}, option_vec=(),
                   highlight_squares=(), ):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.width)
        info.update({
            'box1_is_full': self.box1_is_full,
            'box2_is_full': self.box2_is_full,
        })
        action_names = [o.index(-1) for o in option_vec]
        self.gui.render_board(
            player_position=self.agent_position_idx,
            boxes={
                self.box_positions[0]: _BoxState.CLOSED if self.box1_is_full else _BoxState.OPEN,
                self.box_positions[1]: _BoxState.CLOSED if self.box2_is_full else _BoxState.OPEN,
            },
            walls=self._walls,
            some_matrix=some_matrix,
            policy=policy,
            highlight_square=highlight_square,
            highlight_squares=highlight_squares,
            info=info,
            state_offset=self.num_tiles * (self._hash_state() // self.num_tiles),
            action_names=action_names,
        )

    def __repr__(self):
        # TODO: dirrrty
        return "<BoxWorld instance>"

    def __str__(self):
        # TODO: dirrrty
        return "<BoxWorld instance>"

    @staticmethod
    def get_fitness_fn(SIDE_SIZE):
        TRAINING_NO_CHANGE_STOP = 1000
        GENERATE_RANDOM_OPTIONS = False
        TRAINING_MAX_STEPS = 10000

        TEST_MAX_STEPS_TRAIN = 2000
        TEST_MAX_STEPS_EVAL = 1000
        OPTION_LEARNING_STEPS = 10000

        def fitness_simple_boxes(reward_vector):
            def gen_worlds():
                possible_box_positions = list(itertools.combinations([
                    0,
                    SIDE_SIZE - 1,
                    (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                    SIDE_SIZE * SIDE_SIZE - 1,
                ], 2))
                random.shuffle(possible_box_positions)
                for p in possible_box_positions:
                    yield BoxWorldSimple(side_size=6, box_positions=p)

            possible_worlds = gen_worlds()

            training_world = next(possible_worlds)
            # training_world.show_board()

            sensor_readings = gather_sensor_readings(training_world)

            # define reward fn
            def intrinsic_reward_function(_mdp):
                sensor_reading = sensor_readings[_mdp.agent_position_idx]
                # assert np.allclose(np.sum(reward_vector[sensor_reading]), reward_vector.dot(sensor_reading), 1e-10)
                # return reward_vector.dot(sensor_reading)
                return np.sum(reward_vector[sensor_reading])

            # plot mask activations
            # activactions = []
            # for position_idx in range(training_world.number_of_tiles):
            #     training_world.agent_position_idx = position_idx
            #     activactions.append(intrinsic_reward_function(training_world) > 1)
            # activactions = np.array(activactions)  # .reshape(6, 6)
            # training_world.show_board(highlight_squares=np.argwhere(activactions))

            # generate options set
            learner = learners.q_learning.QLearning(env=training_world, surrogate_reward=intrinsic_reward_function)
            options, cum_reward = learner.learn(max_steps=OPTION_LEARNING_STEPS, generate_options=True)

            # eval options
            cum_cum_reward = 0
            for eval_step, testing_world in enumerate(possible_worlds):
                learner = learners.q_learning.QLearning(env=testing_world, options=options, test_run=True)
                _, _ = learner.learn(max_steps=TEST_MAX_STEPS_TRAIN, generate_options=False)

                cum_reward = learner.test(eval_steps=TEST_MAX_STEPS_EVAL)
                cum_cum_reward += cum_reward
            fitness = cum_cum_reward / eval_step

            print(reward_vector, end="")

            option_names = []
            for option in options:
                for idx, action in enumerate(option):
                    if action == -1:
                        option_names.append(idx)
                        break
            training_world.box_positions = (-1, -1)
            training_world.agent_position_idx = -1
            training_world.show_board(highlight_squares=option_names, info={'score': fitness})
            option_names = " ".join(str(n) for n in sorted(option_names))
            print("score:\t{}\toptions: {}\t{}".format(fitness, len(options), option_names))

            return fitness

        def gather_sensor_readings(training_world):
            sensor_readings = [None, ] * training_world.number_of_tiles
            for position_idx in range(training_world.number_of_tiles):
                reading = np.ones(shape=(3, 3), dtype=np.bool)

                reading[1][1] = 1
                # up down right left
                aaaaa = [
                    ((0, 1), -training_world.width),
                    ((1, 0), -1),
                    # 1, 1 is empty
                    ((1, 1), 1),
                    ((2, 1), training_world.width),
                ]
                for (i, j), d_pos in aaaaa:
                    try:
                        reading[i][j] = (position_idx + d_pos) in training_world._walls[position_idx]
                    except KeyError:
                        pass
                bbbbb = [
                    ((0, 0), (-training_world.width, - 1)),
                    ((0, 2), (-training_world.width, + 1)),
                    ((2, 0), (training_world.width, - 1)),
                    ((2, 2), (training_world.width, + 1)),
                ]
                # can go up
                up_position = position_idx - training_world.width
                up_left_position = position_idx - training_world.width - 1
                up_right_position = position_idx - training_world.width + 1
                if reading[0][1] == 1:
                    if up_position in training_world._walls[up_left_position]:
                        reading[0][0] = 0
                    if up_position in training_world._walls[up_right_position]:
                        reading[0][2] = 0

                left_position = position_idx - 1
                down_left_position = left_position + training_world.width

                if reading[1][0] == 1:
                    if left_position in training_world._walls[up_left_position]:
                        reading[0][0] = 0
                    if left_position in training_world._walls[down_left_position]:
                        reading[2][0] = 0

                down_position = position_idx + training_world.width
                down_right_position = down_position + 1

                if reading[2][1] == 1:
                    if down_position in training_world._walls[down_left_position]:
                        reading[2][0] = 0
                    if down_position in training_world._walls[down_right_position]:
                        reading[2][2] = 0

                right_position = position_idx + 1
                if reading[1][2] == 1:
                    if right_position in training_world._walls[up_right_position]:
                        reading[2][0] = 0
                    if right_position in training_world._walls[down_right_position]:
                        reading[2][2] = 0

                sensor_readings[position_idx] = reading.flatten()
            return sensor_readings

        return fitness_simple_boxes


if __name__ == "__main__":
    import time
    import os
    import timeit

    os.chdir("../")

    test_world = BoxWorldSimple(side_size=6, )
    test_world.show_board = lambda: None

    test_world.reset()
    test_world.show_board()
    time.sleep(0.2)
    test_world.teleport_agent(0)
    test_world.show_board()
    time.sleep(0.2)
    # sequence = [envs.boxes.BoxWorldActions.DOWN] * 6 + \
    #            [envs.boxes.BoxWorldActions.RIGHT] * 6 + \
    #            [envs.boxes.BoxWorldActions.UP] * 6 + \
    #            [envs.boxes.BoxWorldActions.LEFT] * 6

    period = 1 / 60
    timeit.timeit()
    #   def timeit(stmt="pass", setup="pass", timer=default_timer, number=default_number, globals=None):

    """
    for action in sequence:
        test_world.step(action.value)
        test_world.show_board()
        time.sleep(period)
    """

    its = 100000
    for _ in range(int(its / 10)):
        action = random.choice(list(BoxWorldActions))
        test_world.step(action.value)
        # test_world.show_board()
        # time.sleep(period)
    time0 = time.time()
    for _ in range(its):
        action = random.choice(list(BoxWorldActions))
        test_world.step(action.value)
        # test_world.show_board()
        # time.sleep(period)
    diff = (time.time() - time0)
    print("speed", diff)
