import sys
sys.path.append('/Users/seominseok/content/highway-env')

from matplotlib import pyplot as plt
import numpy as np

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import highway_env
from highway_env import *

highway_env.register_highway_envs()

env = gym.make('intersection-merge-v0', render_mode='rgb_array')
env.reset()
for _ in range(5):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()