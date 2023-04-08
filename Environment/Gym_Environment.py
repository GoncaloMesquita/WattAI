import gym
from gym import spaces
import numpy as np


class MLPEnvironment(gym.Env):
    def __init__(self, your_mlp_function, state_dim, action_dim, action_bound):
        super(MLPEnvironment, self).__init__()

        # Set your custom MLP function here
        self.your_mlp_function = your_mlp_function

        # Define action and observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32) # Falta definir as bounds, olhar para os max e min de cada coluna do dataset
        
        # Adjust the action_space shape to handle multiple continuous actions
        self.action_space = spaces.Box(low=-action_bound, high=action_bound, shape=(action_dim,), dtype=np.float32) # Falta definir as bounds

    def step(self, action, state, models_dict, scalers_dict):
        # Implement your step function using the custom MLP function
        next_state, reward, done, info = self.your_mlp_function(action, state, models_dict, scalers_dict)
        return next_state, reward, done, info

    def reset(self):
        # Implement reset function to return the initial state of the environment
        initial_state = np.random.rand(self.observation_space.shape[0])
        return initial_state

    def render(self, mode='human'):
        # Optionally, implement the render function for visualization
        pass

    def close(self):
        # Optionally, implement the close function to clean up resources
        pass
