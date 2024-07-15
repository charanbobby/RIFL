import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from gym.wrappers import GrayScaleObservation
from matplotlib import pyplot as plt
#Import Frame Stacker Wrapper and Grayscale Observation Wrapper 
# #(to know where Mario is moving in, what velocity he is moving in, etc.)
# #(Grayscale Observation Wrapper is used to convert the RGB image to Grayscale to have less data to process)
#Originally built by OPENAI, but now maintained by Stable Baselines - will be using PPO (Proximal Policy Optimization) algorithm
# # (VecFrameStack is used to stack the frames of the game to have a better understanding of the game)
# # (DummyVecEnv is used to create a vectorized environment to run multiple environments in parallel)
#Importing Matplotlib to show the impact of frame stacking

# Import os for file path management
import os

# Import PPO (Proximal Policy Optimization) for algorithm 
from stable_baselines3 import PPO 

# Import Base Callback for saving models (save every 10,000 steps - will not loose the model's progress)
from stable_baselines3.common.callbacks import BaseCallback

class TrainingLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainingLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Setup the call back
callback = TrainingLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)


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

#6. Create the model 
# # Policy network is CNN (Convolutional Neural Network) - Brain of the AI - CNN is fast for processing images
# # MlpPolicy is used to predict the probability of the action to be taken (good for excel)
# # LSTM Policy is used to predict the probability of the action to be taken (good for time series data)
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001,n_steps=512)

#Train the AI model learn from the environment
model.learn(total_timesteps=10000000, callback=callback)

#state = env.reset()

done = True
for step in range(5000):
    if done:
        state = env.reset()
    
    state, reward, done, info = env.step([env.action_space.sample()])  # take a random action
    env.render()

env.close()