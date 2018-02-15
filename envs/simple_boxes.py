import enum
import disk_utils
import collections
import numpy as np
import itertools
import envs.gridworld
import envs.gui
import learners
import random
import gym.spaces
import learners.q_learning
import matplotlib.pyplot as plt


class BoxWorldActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class _BoxState(enum.Enum):
    OPEN = 0
    CLOSED = 2


class BoxWorldSimple(envs.gridworld.GridWorld):
    HACK = 0

    def __init__(self, side_size, box_positions=(), agent_lifetime=5):
        super(BoxWorldSimple, self).__init__(side_size, terminal_states=(), base_transition_probability=0.9,)
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
        if len(self.box_positions) > 0:
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
        if len(self.box_positions) > 0:
            boxes_state = {
                self.box_positions[0]: _BoxState.CLOSED if self.box1_is_full else _BoxState.OPEN,
                self.box_positions[1]: _BoxState.CLOSED if self.box2_is_full else _BoxState.OPEN,
            }
        else:
            boxes_state = {}

        self.gui.render_board(
            player_position=self.agent_position_idx,
            boxes=boxes_state,
            walls=self._walls,
            some_matrix=some_matrix,
            policy=policy,
            highlight_square=highlight_square,
            highlight_squares=highlight_squares,
            info=info,
            state_offset=self.number_of_tiles * (self._hash_state() // self.number_of_tiles),
            action_names=action_names,
        )

    def __repr__(self):
        return "<BoxWorld {} instance>".format(self.number_of_tiles)

    def __str__(self):
        # TODO: dirrrty
        return "<BoxWorld {} instance>".format(self.number_of_tiles)

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
            possible_box_positions = list(itertools.combinations([
                0,
                SIDE_SIZE - 1,
                (SIDE_SIZE * SIDE_SIZE) - SIDE_SIZE,
                SIDE_SIZE * SIDE_SIZE - 1,
            ], 2))
            random.shuffle(possible_box_positions)
            xs = [2, 16, 32, 64, 128, 256, 512, 1024, 2048, 10000]
            dxs = [xs[0], ] + [x - xs[idx] for idx, x in enumerate(xs[1:])]

            cum_scores = collections.defaultdict(float)
            for eval_step, box_positions in enumerate(possible_box_positions):
                option_set_scores = BoxWorldSimple.eval_option_on_mdp(TEST_MAX_STEPS_EVAL, box_positions, options, dxs)
                for k in option_set_scores.keys():
                    cum_scores[k] += option_set_scores[k]
            fitness = cum_cum_reward / eval_step

            print(reward_vector, end="")

            option_names = []
            for option in options:
                for idx, action in enumerate(option):
                    if action == -1:
                        option_names.append(idx)
                        break

            BoxWorldSimple.HACK += 1
            if BoxWorldSimple.HACK % 10 == 0:
                training_world.box_positions = (-1, -1)
                training_world.agent_position_idx = -1
                training_world.show_board(highlight_squares=option_names, info={'score': fitness})
            option_names = " ".join(str(n) for n in sorted(option_names))
            print("score:\t{}\toptions: {}\t{}".format(fitness, len(options), option_names))

            return fitness

        def gather_sensor_readings(training_world):
            world_tiles = training_world.number_of_tiles
            world_walls = training_world._walls
            world_width = training_world.width

            sensor_readings = [None, ] * world_tiles

            for position_idx in range(world_tiles):
                # tmp_world.show_board(highlight_square=position_idx)
                can_reach = np.ones(shape=(3, 3), dtype=np.bool) * False

                can_reach[1][1] = True

                # can go up
                up_position = position_idx - world_width
                up_left_position = position_idx - world_width - 1
                up_right_position = position_idx - world_width + 1
                left_position = position_idx - 1
                down_left_position = left_position + world_width
                down_position = position_idx + world_width
                down_right_position = down_position + 1
                right_position = position_idx + 1

                up_coord = (0, 1)
                up_right_coord = (0, 2)
                right_coord = (1, 2)
                down_right_coord = (2, 2)
                down_coord = (2, 1)
                down_left_coord = (2, 0)
                left_coord = (1, 0)
                up_left_coord = (0, 0)

                if up_position not in world_walls[position_idx]:
                    can_reach[up_coord] = True
                if left_position not in world_walls[position_idx]:
                    can_reach[left_coord] = True
                if down_position not in world_walls[position_idx]:
                    can_reach[down_coord] = True
                if right_position not in world_walls[position_idx]:
                    can_reach[right_coord] = True

                if can_reach[up_coord] and can_reach[left_coord]:
                    if up_left_position not in world_walls[left_position].union(world_walls[up_position]):
                        can_reach[up_left_coord] = True

                if can_reach[down_coord] and can_reach[left_coord]:
                    if down_left_position not in world_walls[left_position].union(world_walls[down_position]):
                        can_reach[down_left_coord] = True

                if can_reach[down_coord] and can_reach[right_coord]:
                    if down_right_position not in world_walls[right_position].union(world_walls[down_position]):
                        can_reach[down_right_coord] = True

                if can_reach[up_coord] and can_reach[right_coord]:
                    if up_right_position not in world_walls[right_position].union(world_walls[up_position]):
                        can_reach[up_right_coord] = True

                # if position_idx > 5:
                #     plt.matshow(can_reach.reshape((3, 3)))
                #     plt.show()

                sensor_readings[position_idx] = np.logical_not(can_reach).flatten()
                # sensor_readings[position_idx] = can_reach.flatten()
            return sensor_readings

        return fitness_simple_boxes

    @classmethod
    @disk_utils.disk_cache
    def eval_option_on_mdp(cls, SIDE_SIZE, box_positions, option_vec, xs):
        mdp = cls(side_size=SIDE_SIZE, box_positions=box_positions)
        learner = learners.q_learning.QLearning(env=mdp, options=option_vec)
        _, _, fitnesses = learner.learn(xs=xs)
        return fitnesses


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
