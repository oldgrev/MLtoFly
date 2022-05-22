from stable_baselines3.common.env_checker import check_env
from rlenv import flyEnv
import numpy as np


env = flyEnv()
#print(f'Required shape: {env.observation_space.shape}')
#print(f'low: {env.observation_space.low}')
#print(f'high: {env.observation_space.high}')
# It will check your custom environment and output additional warnings if needed
check_env(env)