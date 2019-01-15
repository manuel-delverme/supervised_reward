import unittest
import envs.hungry_thirsty
import envs.gridworld


class TestGridWorld(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)

    def test_env(self):
        test_world = envs.hungry_thirsty.HungryThirsty(side_size=7, )
        self.assertTrue(True)

    def test_movement(self):
        test_world = envs.hungry_thirsty.HungryThirsty(side_size=7, )
        test_world.reset()
        sequence = [envs.hungry_thirsty.HungryThirstyActions.DOWN] * 6 + \
                   [envs.hungry_thirsty.HungryThirstyActions.RIGHT] * 6 + \
                   [envs.hungry_thirsty.HungryThirstyActions.UP] * 6 + \
                   [envs.hungry_thirsty.HungryThirstyActions.LEFT] * 6 + \
                   [envs.hungry_thirsty.HungryThirstyActions.EAT_FOOD] * 6 + \
                   [envs.hungry_thirsty.HungryThirstyActions.DRINK_WATER] * 6

        for action in sequence:
            transition = test_world.step(action)
        self.assertTrue(True)

    def test_render(self):
        test_world = envs.hungry_thirsty.HungryThirsty(side_size=7, )
        test_world.reset()
        ascii_art = test_world.render(mode='ascii')
        print(ascii_art)
