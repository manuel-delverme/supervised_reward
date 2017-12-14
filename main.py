import gym
import numpy as np
import sys
import envs.gridworld

env = envs.gridworld.GridWorld()
terminal = False
while not terminal:
    state, reward, terminal, info = env.step(1)
    env.render(mode="ansi")
