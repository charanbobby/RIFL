import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from gym.wrappers import GrayScaleObservation
from matplotlib import pyplot as plt
import os

# Import PPO (Proximal Policy Optimization) for algorithm 
from stable_baselines3 import PPO 

# Import Base Callback for saving models (save every 10,000 steps - will not loose the model's progress)
from stable_baselines3.common.callbacks import BaseCallback

#1. Create the environment
env = gym_super_mario_bros.make('SuperMarioBros2-v1')

#2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)

#4. Wrap inside the Dummy environment
env = DummyVecEnv([lambda: env])

#5 Stack the frames - 4 frames are stacked
env = VecFrameStack(env, 4, channels_order='last')

#6. Load the pre-trained model
model = PPO.load('./train/best_model_100000')

#7. Play the game
state = env.reset()

# Loop through the game
while True: 
    action, _state = model.predict(state)   
    state, reward, done, info = env.step(action)  # take a random action
    env.render()