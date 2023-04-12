import gym
from gym import spaces
import numpy as np
import pandas as pd


class MLPEnvironment(gym.Env):
    def __init__(self, your_mlp_function, state_dim, action_dim):
        super(MLPEnvironment, self).__init__()

        # Set your custom MLP function here
        self.your_mlp_function = your_mlp_function

        # Define action and observation space
        self.observation_space = spaces.Box(low=np.array([0,16,350,9,9,0]), high=np.array([42,30,1000,30,30,40000]), shape=(state_dim,), dtype=np.float64) # Falta definir as bounds, olhar para os max e min de cada coluna do dataset
        
        # Adjust the action_space shape to handle multiple continuous actions
        self.action_space = spaces.Box(low=np.array([20.14,9.40,0,0,0]), high=np.array([29.35,24,100,100,100]), shape=(action_dim,), dtype=np.float64) # Falta definir as bounds

        self.day = 96/2

    def step(self, action, state, models_dict, scalers_dict,next_outdoor_temp):
        # Implement your step function using the custom MLP function
        next_state, reward = self.your_mlp_function(action, state, models_dict, scalers_dict,next_outdoor_temp)

        self.day -= 1
        if self.day == 0:
            done = True
        else:
            done = False

        info = {}
        return next_state, reward, done, info

    def reset(self, data_set):
        # np.random.seed(0)
        # choose random row from dataset and retrieve index of row
        index = np.random.randint(0, data_set.shape[0])

        # Implement reset function to return the initial state of the environment
        initial_state = data_set.iloc[index,[1,4,5,8,9,10]].values
        self.day = 96/2
        
        
        return initial_state, index

    def render(self, mode='human'):
        # Optionally, implement the render function for visualization
        pass

    def close(self):
        # Optionally, implement the close function to clean up resources
        pass
