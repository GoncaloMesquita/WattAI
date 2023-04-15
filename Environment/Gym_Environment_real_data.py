import gym
from gym import spaces
import numpy as np
import pandas as pd


class Real_Environment(gym.Env):
    def __init__(self, real_env, state_dim, action_dim, data_set):
        super(Real_Environment, self).__init__()

        self.real_env = real_env

        # Define action and observation space
        self.observation_space = spaces.Box(low=np.array([0,16,350,9,9,0]), high=np.array([42,30,1000,30,30,40000]), shape=(state_dim,), dtype=np.float64) # Falta definir as bounds, olhar para os max e min de cada coluna do dataset
        
        # Adjust the action_space shape to handle multiple continuous actions
        self.action_space = spaces.Box(low=np.array([20.14,9.40,0,0,0]), high=np.array([29.35,24,100,100,100]), shape=(action_dim,), dtype=np.float64) # Falta definir as bounds

        self.day = 96

        self.data_set = data_set
    def step(self, action, index):
        # Implement your step function using the real environment
        next_state, reward = self.real_env(action, self.data_set, index)

        self.day -= 1
        if self.day == 0:
            done = True
        else:
            done = False

        info = {}
        return next_state, reward, done, info

    def reset(self):
        # np.random.seed(0)
        # choose random row from dataset and retrieve index of row
        index = np.random.randint(0, self.data_set.iloc[:-97].shape[0])

        # Implement reset function to return the initial state of the environment
        initial_state = self.data_set.iloc[index,[1,4,5,8,9,10]].values
        self.day = 96
        
        
        return initial_state, index

    def render(self, mode='human'):
        # Optionally, implement the render function for visualization
        pass

    def close(self):
        # Optionally, implement the close function to clean up resources
        pass
