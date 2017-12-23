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
    def __init__(self, side_size, box_positions=(0, 5, 30, 35), num_boxes=2):
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
        self.water_position, self.food_position = box_positions
        self.observation_space = gym.spaces.Discrete(self.observation_space.n * extra_dof)
        self.action_space = gym.spaces.Discrete(self.action_space.n + len(self._state))

    def _step(self, action):
        action = HungryThirstyActions(action)
        self._state['hungry'] = True
        terminal = False
        info = {}

        if random.random() < 0.1:
            self._state['thirsty'] = True

        if action == HungryThirstyActions.EAT_FOOD:
            if self.agent_position_idx == self.food_position and not self._state['thirsty']:
                self._state['hungry'] = False
        elif action == HungryThirstyActions.DRINK_WATER:
            if self.agent_position_idx == self.water_position:
                self._state['thirsty'] = False
        else:
            tile_idx, reward, terminal, info = super(envs.gridworld.GridWorld, self)._step(action.value)

        state_obj = self._hash_state()

        reward = 0
        if not self._state['hungry']:
            reward += 1
        else:
            reward += -1

        if not self._state['thirsty']:
            reward += 0.1
        else:
            reward += -1

        return state_obj, reward, terminal, info

    def _hash_state(self):
        state_hash = self.agent_position_idx
        state_hash += self._state['hungry'] * self.number_of_tiles
        state_hash += self._state['thirsty'] * self.number_of_tiles * 2
        return state_hash  # state.State(state_hash=state_hash, state_info=self._state.copy())

    def print_board(self, some_matrix=None, close=False, policy=None):
        if close:
            return
        if self.gui is None:
            self.gui = envs.gui.GUI(self.width)
        self.gui.print_board(
            player_position=self.agent_position_idx,
            player_state=self._state,
            terminal_states=self.terminal_positions,
            walls=self._walls, boxes=self.boxes,
            some_matrix=some_matrix,
            policy=policy,
        )

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
