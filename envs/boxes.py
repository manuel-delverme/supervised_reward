import enum
import envs.gridworld
import random
import gym.spaces


class BoxWorldActions(enum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    OPEN_BOX = 4
    EAT_FOOD = 5


class _BoxState(enum.Enum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 1


class BoxWorld(envs.gridworld.GridWorld):
    def __init__(self, side_size, box_positions=(0, 30)):
        super(BoxWorld, self).__init__(
            side_size=side_size,
            terminal_states=(),
            base_transition_probability=0.9,
        )
        self._state = {
            'hungry': True,
            'box': {},
        }
        for box_position in box_positions:
            self._state['box'][box_position] = _BoxState.CLOSED
        self.box_positions = box_positions

        extra_dof = len(self._state['box']) * len(_BoxState) * 2  # box states and 2 for hungry in True, False
        self.observation_space = gym.spaces.Discrete(self.observation_space.n * extra_dof)
        self.action_space = gym.spaces.Discrete(self.action_space.n + len(self._state))

    def _step(self, action):
        action = BoxWorldActions(action)
        self._state['hungry'] = True
        terminal = False
        info = {}

        in_a_box = False
        for box_pos in self.box_positions:
            distance = box_pos - self.agent_position_idx
            if distance in (0, -1, +1, -self.width, self.width):
                in_which_box = box_pos
                in_a_box = True
                # MINIMUM BOX DISTANCE == 2 OR bugS
                break

        # boxes stay half open only for a timestep
        for box_pos in self.box_positions:
            if self._state['box'][box_pos] == _BoxState.HALF_OPEN:
                self._state['box'][box_pos] = _BoxState.OPEN

        # open boxes randomly close
        for box_pos in self.box_positions:
            if random.random() < 0.1 and self._state['box'][box_pos] == _BoxState.OPEN:
                self._state['box'][box_pos] = _BoxState.CLOSED

        if action == BoxWorldActions.EAT_FOOD:
            if in_a_box and self._state['box'][in_which_box] == _BoxState.HALF_OPEN:
                self._state['hungry'] = False
        elif action == BoxWorldActions.OPEN_BOX:
            if in_a_box:
                self._state['box'][in_which_box] = _BoxState.HALF_OPEN
        else:
            tile_idx, reward, terminal, info = super(envs.gridworld.GridWorld, self)._step(action.value)

        state_obj = self._hash_state()

        if not self._state['hungry']:
            reward = 100
            terminal = True
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

    def print_board(self, some_matrix=None, close=False, policy=None, highlight_square=None):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.width)
        self.gui.print_board(
            player_position=self.agent_position_idx,
            terminal_states=self.terminal_positions,
            boxes=self.box_positions,
            walls=self._walls,
            thirsty=False,  # there is no thirsty in boxes
            hungry=True,
            some_matrix=some_matrix,
            policy=policy,
            highlight_square=highlight_square,
        )

    def force_state(self, state):
        #TODO: remove, will cause bugs
        super(BoxWorld, self).teleport_agent(state % self.num_tiles)

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
