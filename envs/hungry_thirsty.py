import enum
import envs.gridworld
import random
import gym.spaces


# FUCK MY LIFE
class HungryThirstyActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    DRINK_WATER = 4
    EAT_FOOD = 5


# HungryThirstyActions = OPEN_BOX = 4


class HungryThirsty(envs.gridworld.GridWorld):
    def __init__(self, side_size, water_position=0, food_position=5):
        self._state = {
            'hungry': True,
            'thirsty': True,
        }
        # action_size = 6
        # state_size = board_size * 2 * len(self._state)
        # for action in new_actions:
        #     # overwrite actions
        #     # transition[position_idx][Actions.OPEN_BOX] = [(1.0, position_idx, reward, False)]
        #     transition[position_idx][Actions.EAT_FOOD] = [(1.0, position_idx, reward, False)]
        #     transition[position_idx][Actions.DRINK_WATER] = [(1.0, position_idx, reward, False)]
        super(HungryThirsty, self).__init__(
            side_size=side_size,
            terminal_states=(),
            base_transition_probability=0.9,
        )
        extra_dof = len(self._state) * 2
        self.water_position, self.food_position = water_position, food_position
        self.observation_space = gym.spaces.Discrete(self.observation_space.n * extra_dof)
        self.action_space = gym.spaces.Discrete(self.action_space.n + len(self._state))

    def _step(self, action):
        action = HungryThirstyActions(action)
        terminal = False
        info = {}
        self._state['hungry'] = True

        if action == HungryThirstyActions.EAT_FOOD:
            if self.agent_position_idx == self.food_position and not self._state['thirsty']:
                self._state['hungry'] = False
        elif action == HungryThirstyActions.DRINK_WATER:
            if self.agent_position_idx == self.water_position:
                self._state['thirsty'] = False
        else:
            tile_idx, reward, terminal, info = super(envs.gridworld.GridWorld, self)._step(action.value)

        state_obj = self._hash_state()

        reward = -1
        if not self._state['hungry']:
            reward = 100
            self._state['thirsty'] = True

        # else:
        #     reward += -1

        # if not self._state['thirsty']:
        #     reward += 0.1
        # else:
        #     reward += -1

        # if random.random() < 0.001:
        #     self._state['thirsty'] = True

        return state_obj, reward, terminal, info

    def _reset(self):
        _ = super(envs.gridworld.GridWorld, self)._reset()
        self._state['hungry'] = True
        self._state['thirsty'] = random.choice((True, False))
        return self._hash_state()

    def _hash_state(self):
        state_hash = self.agent_position_idx
        state_hash += self._state['hungry'] * self.number_of_tiles
        state_hash += self._state['thirsty'] * self.number_of_tiles * 2
        return state_hash  # state.State(state_hash=state_hash, state_info=self._state.copy())

    def show_board(self, some_matrix=None, close=False, policy=None, highlight_square=None):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.width)
        info = self._state.copy()
        self.gui.render_board(
            player_position=self.agent_position_idx,
            terminal_states=self.terminal_positions,
            water_position=self.water_position,
            food_position=self.food_position,
            walls=self._walls,
            thirsty=False,  # there is no thirsty in boxes
            hungry=True,
            some_matrix=some_matrix,
            policy=policy,
            highlight_square=highlight_square,
            info=info,
            state_offset=self.num_tiles * (self._hash_state() // self.num_tiles)
        )

    def force_state(self, state):
        position, state_dict = self._decode_state(state)
        self._state = state_dict
        super(HungryThirsty, self).teleport_agent(position)

    def _decode_state(self, state_hash):
        # state = pos + hungry * tiles + thirsty * tiles * 2
        state_dict = {}
        if state_hash >= self.number_of_tiles * 2:
            state_hash -= self.number_of_tiles * 2
            state_dict['thirsty'] = True
        else:
            state_dict['thirsty'] = False

        if state_hash >= self.number_of_tiles:
            state_dict['hungry'] = True
            state_hash -= self.number_of_tiles
        elif state_hash < self.number_of_tiles:
            state_dict['hungry'] = False
        return state_hash, state_dict

    def __repr__(self):
        # TODO: dirrrty
        return "<BoxWorld instance>"

    def __str__(self):
        # TODO: dirrrty
        return "<BoxWorld instance>"

    @staticmethod
    def get_env_fitness_fn(SIDE_SIZE):
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
        return fitness_hungry_thirsty, 4


if __name__ == "__main__":
    import time

    test_world = HungryThirsty(side_size=6, )
    test_world.reset()
    test_world.render()
    time.sleep(0.2)
    test_world.teleport_agent(0)
    test_world.render()
    time.sleep(0.2)
    sequence = [envs.gridworld.GridWorldActions.DOWN] * 6 + \
               [envs.gridworld.GridWorldActions.RIGHT] * 6 + \
               [envs.gridworld.GridWorldActions.UP] * 6 + \
               [envs.gridworld.GridWorldActions.LEFT] * 6 + \
               [envs.gridworld.GridWorldActions.EAT_FOOD] * 6 + \
               [envs.gridworld.GridWorldActions.DRINK_WATER] * 6

    period = 1 / 60
    for action in sequence:
        test_world.step(action)
        test_world.render()
        time.sleep(period)
    while True:
        action = random.choice(list(envs.gridworld.GridWorldActions))
        test_world.step(action)
        test_world.render()
        time.sleep(period)
