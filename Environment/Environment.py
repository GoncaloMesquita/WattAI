import numpy as np
from tensorflow import keras
import pandas as pd
import torch
import sys
import warnings

def Environment(action, state, models_dict, scalers_dict, next_outdoor_temp):
    """
    action: action to be taken by the agent (array)
    state: current state of the environment (array)
    models_dict: dictionary with all the models
    scalers_dict: dictionary with all the scalers
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Environment evolution
    environment = np.array([np.concatenate((action, state), axis=0)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        environment = scalers_dict['scaler_environment'].transform(environment)
    indoor_temp_co2 = models_dict['model_next_state'](torch.tensor(environment).float())
    air_temp_suplly_return = models_dict['model_air_temp_suplly_return'](torch.tensor(environment).float())     
    air_flowrate = models_dict['model_air_flowrate'](torch.tensor(environment).float())
    
    
    next_state = np.concatenate([np.array([[next_outdoor_temp]]), indoor_temp_co2.detach_().numpy(), air_temp_suplly_return.detach_().numpy(), air_flowrate.detach_().numpy()], axis=1)
    
    # Energy prediction
    features_energy = np.concatenate([next_state[:,:1],np.array([action[:2]]),next_state[:,1:3],np.array([action[2:4]]),next_state[:,3:],np.array([action[-1:]])], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled_features = scalers_dict['augmented_data_scaler'].transform(features_energy)    
    energy = models_dict['energy_model_augmented_data'].predict(scaled_features, verbose=0)
    
    # Comfort prediction
    std_T = 2
    thermal_comfort = np.exp(-(next_state[0,1]-21)**2/(2*std_T**2))/(std_T*np.sqrt(2*np.pi))
    CO2_comfort = -1/(1+np.exp(-0.2*(next_state[0,2]-700)))+1

    max_thermal_comfort = 0.199
    if energy[0,0]<0:
        energy[0,0] = -energy[0,0].copy()
    # Reward calculation
    reward = -.5*energy[0,0]/130 + .35*thermal_comfort/max_thermal_comfort + .15*CO2_comfort

    final_reward = 1/torch.clamp(torch.tensor(abs(reward-0.5)), min=0.00000001, max=abs(reward.copy()))

    return next_state[0], final_reward