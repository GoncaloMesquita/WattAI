import numpy as np
from tensorflow import keras
import pandas as pd
import torch
import sys
import warnings

def Environment_real_data(action, real_data, index):

    # next_state = real_data.iloc[index,[1,12,13,14,15,16]]
    
    next_state = np.array([real_data.iloc[index,[1,12,13,14,15,16]].values])
    real_action = np.array([real_data.iloc[index,[2,3,6,7,11]].values])
    energy = real_data.iloc[index+1,0]
    # reward
    std_T = 2
    thermal_comfort = np.exp(-(next_state[0,1]-21)**2/(2*std_T**2))/(std_T*np.sqrt(2*np.pi))
    CO2_comfort = -1/(1+np.exp(-0.2*(next_state[0,2]-700)))+1
    max_thermal_comfort = 0.199

    reward = -.5*energy/130 + .35*thermal_comfort/max_thermal_comfort + .15*CO2_comfort


    final_reward = 1/torch.clamp(torch.tensor(abs(reward-0.5)), min=0.00000001, max=abs(reward.copy()))

    return next_state[0],real_action, final_reward