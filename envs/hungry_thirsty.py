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
