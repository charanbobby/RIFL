# Import vizdoom for game env
from vizdoom import *
# Import radom action for sampling random actions
import random
# Import time for sleep
import time
# Import numpy for identity matrix
import numpy as np
# Import environment base class from OpenAI gym
from gym import Env
# Import gym spaces for defining the action space, Box returns a n-dim array, Discrete returns a single value
from gym.spaces import Discrete, Box
# Import opencv for image processing (grayscaling)
import cv2

# Create VizDoom OPENAI Gym Environment
class VizDoomEnv(Env):

    # Function to initialize the environment
    def __init__(self, render = False):
        # Create a game instance
        self.game = DoomGame()
        # Load the configuration file
        self.game.load_config(r'.\Vizdoom\gtihub\ViZDoom\scenarios\basic.cfg')
        # Define the action space
        self.action_space = Discrete(3)
        # Define the observation space
        self.observation_space = Box(low = 0, high = 255, shape = (3, 240, 320), dtype=np.uint8)
        
        # Render the game only when we want to visualize it
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        # Initialize the game
        self.game.init()

    # Function to take a step in the environment
    def step(self, action):
        # Specify the actions and take the action
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action], 4) #4 frames per action
        
        # Get the other parameters that we need to return
        if self.game.get_state():  #handling the case when the game is finished
            state = self.game.get_state().screen_buffer
            info = state.game.get_state().game_variables
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        done = self.game.is_episode_finished()   

        return state, reward, done, info
    
    def render():
        pass
    # Function to reset the environment when we start a new game
    def reset():
        pass
    # Grayscale the game frame and resize it
    def grayscale():
        pass

    # Call to close the game
    def close(self):
        self.game.close()
        pass


# Setup the game
env = VizDoomEnv(render=True)
env.step(1)
env.close()
# game.load_config(r'.\Vizdoom\gtihub\ViZDoom\scenarios\basic.cfg')
# Set of actions that we can take

# # Function to take random action
# random.choice(actions) 

# # Loop to play the game
# episodes = 10
# for episode in range(episodes):
#     # Create a new episode
#     game.new_episode()
#     # Check if the game is not finished
#     while not game.is_episode_finished():
#         state = game.get_state() # game frame
#         img = state.screen_buffer # image frame
#         misc = state.game_variables # available game variables from config file
#         reward = game.make_action(random.choice(actions),4)
#         print('\treward:', reward)
#         time.sleep(0.02)
#     print('Result:', game.get_total_reward())
#     time.sleep(2)

# game.close()