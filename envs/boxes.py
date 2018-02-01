import enum
import envs.gridworld
import random
import gym.spaces
import itertools


class BoxWorldActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    OPEN_BOX = 4
    EAT_FOOD = 5


class _BoxState(enum.Enum):
    OPEN = 0
    HALF_OPEN = 1
    CLOSED = 2


class BoxWorld(envs.gridworld.GridWorld):
    def __init__(self, side_size, box_positions=(0, 30)):
        super(BoxWorld, self).__init__(
            side_size=side_size,
            terminal_states=(),
            base_transition_probability=0.9,
        )
        self._state = {
            'hungry': random.choice((True, False)),
            'box': {},
        }
        for box_position in box_positions:
            self._state['box'][box_position] = random.choice(list(_BoxState))
        self.box_positions = box_positions

        extra_dof = (len(_BoxState) ** len(self._state['box'])) * 2  # box states and 2 for hungry in True, False
        self.observation_space = gym.spaces.Discrete(self.observation_space.n * extra_dof)
        self.action_space = gym.spaces.Discrete(len(BoxWorldActions))

    def _step(self, action):
        action = BoxWorldActions(action)
        self._state['hungry'] = True
        terminal = False
        info = {}

        in_a_box = False
        for box_pos in self.box_positions:
            distance = box_pos - self.agent_position_idx
            if distance == 0:
                in_which_box = box_pos
                in_a_box = True

            # if distance in (0, -1, +1, -self.width, self.width):
            #     in_which_box = box_pos
            #     in_a_box = True
            #     # MINIMUM BOX DISTANCE == 2 OR bugS
            #     break

        if action == BoxWorldActions.EAT_FOOD:
            if in_a_box and self._state['box'][in_which_box] == _BoxState.HALF_OPEN:
                self._state['hungry'] = False

        # open boxes randomly close
        for box_pos in self.box_positions:
            if random.random() < 0.1 and self._state['box'][box_pos] == _BoxState.OPEN:
                self._state['box'][box_pos] = _BoxState.CLOSED

        # boxes stay half open only for a timestep
        for box_pos in self.box_positions:
            if self._state['box'][box_pos] == _BoxState.HALF_OPEN:
                self._state['box'][box_pos] = _BoxState.OPEN

        if action == BoxWorldActions.OPEN_BOX:
            if in_a_box:
                self._state['box'][in_which_box] = _BoxState.HALF_OPEN

        if action not in (BoxWorldActions.OPEN_BOX, BoxWorldActions.EAT_FOOD):
            tile_idx, reward, terminal, info = super(envs.gridworld.GridWorld, self)._step(action.value)

        state_obj = self._hash_state()

        if not self._state['hungry']:
            reward = 100
        else:
            reward = -1

        return state_obj, reward, terminal, info

    def _hash_state(self):
        offset = 1
        state_hash = self.agent_position_idx * offset
        offset *= self.number_of_tiles
        state_hash += self._state['hungry'] * offset
        offset *= len((True, False))
        for pos, state in self._state['box'].items():
            state_hash += state.value * offset
            offset *= len(_BoxState)
        return state_hash  # state.State(state_hash=state_hash, state_info=self._state.copy())

    def reset(self):
        super(BoxWorld, self).reset()
        self._state['hungry'] = random.choice((True, False))
        for pos in self._state['box'].keys():
            self._state['box'][pos] = random.choice(list(_BoxState))
        return self._hash_state()

    def show_board(self, some_matrix=None, close=False, policy=None, highlight_square=None, just_numbers=True):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.width)
        info = self._state.copy()
        self.gui.render_board(
            player_position=self.agent_position_idx,
            terminal_states=self.terminal_positions,
            boxes=self._state['box'],
            walls=self._walls,
            thirsty=False,  # there is no thirsty in boxes
            hungry=True,
            some_matrix=some_matrix,
            policy=policy,
            highlight_square=highlight_square,
            info=info,
            state_offset=self.num_tiles * (self._hash_state() // self.num_tiles),
            just_numbers=True
        )

    def force_state(self, state):
        # TODO: remove, will cause bugs
        super(BoxWorld, self).teleport_agent(state % self.num_tiles)

    @staticmethod
    def get_fitness_fn(SIDE_SIZE):
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

        return fitness_boxes

if __name__ == "__main__":
    import time

    test_world = BoxWorld(side_size=6, )
    test_world.reset()
    test_world.render()
    time.sleep(0.2)
    test_world.teleport_agent(0)
    test_world.render()
    time.sleep(0.2)
    sequence = [envs.boxes.BoxWorldActions.DOWN] * 6 + \
               [envs.boxes.BoxWorldActions.RIGHT] * 6 + \
               [envs.boxes.BoxWorldActions.UP] * 6 + \
               [envs.boxes.BoxWorldActions.LEFT] * 6

    period = 1 / 60
    for action in sequence:
        test_world.step(action.value)
        test_world.render()
        time.sleep(period)
    while True:
        action = random.choice(list(envs.boxes.BoxWorldActions))
        test_world.step(action.value)
        test_world.render()
        time.sleep(period)
