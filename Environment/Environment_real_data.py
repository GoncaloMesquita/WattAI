import numpy as np
from tensorflow import keras
import pandas as pd
import torch
import sys
import warnings

def Environment_real_data(action, real_data, index):

    # next_state = real_data.iloc[index,[1,12,13,14,15,16]]
    
    next_state = np.array([real_data.iloc[index,[1,12,13,14,15,16]]])
    real_action = np.array([real_data.iloc[index,[2,3,6,7,11]].values])

    # reward

    

    reward = np.linalg.norm(action-real_action[0])**2

    final_reward = 1/torch.clamp(torch.tensor(reward), min=0.00000001, max=reward.copy())

    return next_state[0], final_reward