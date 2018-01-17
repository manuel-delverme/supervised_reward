import enum
import envs.gridworld
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
    def __init__(self, side_size, box_positions=(0, 30)):
        super(BoxWorldSimple, self).__init__(
            side_size=side_size,
            terminal_states=(),
            base_transition_probability=0.9,
        )
        self.box1_is_full = None
        self.box2_is_full = None
        self.box_positions = box_positions

        self.reset()

        extra_dof = (len(_BoxState) ** 2)
        self.observation_space = gym.spaces.Discrete(self.observation_space.n * extra_dof)
        self.action_space = gym.spaces.Discrete(len(BoxWorldActions))

    def _step(self, action):
        tile_idx, gridworld_reward, terminal, info = super(envs.gridworld.GridWorld, self)._step(action)

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
        self.box1_is_full = True
        self.box2_is_full = True
        return self._hash_state()

    def show_board(self, some_matrix=None, close=False, policy=None, highlight_square=None):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.width)
        info = {
            'box1_is_full': self.box1_is_full,
            'box2_is_full': self.box2_is_full,
        }
        self.gui.render_board(
            player_position=self.agent_position_idx,
            boxes={
                self.box_positions[0]: self.box1_is_full,
                self.box_positions[1]: self.box2_is_full,
            },
            walls=self._walls,
            some_matrix=some_matrix,
            policy=policy,
            highlight_square=highlight_square,
            info=info,
            state_offset=self.num_tiles * (self._hash_state() // self.num_tiles)
        )

    def __repr__(self):
        # TODO: dirrrty
        return "<BoxWorld instance>"

    def __str__(self):
        # TODO: dirrrty
        return "<BoxWorld instance>"

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
