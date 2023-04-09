import numpy as np
from tensorflow import keras
import pandas as pd
import torch
import sys
sys.path.append('Comfort_model')
import comfort_predictor

def Environment(action, state, models_dict, scalers_dict, next_outdoor_temp):
    """
    action: action to be taken by the agent (array)
    state: current state of the environment (array)
    models_dict: dictionary with all the models
    scalers_dict: dictionary with all the scalers
    """
    # Environment evolution
    environment = np.array([np.concatenate((action, state), axis=0)])
    environment = scalers_dict['scaler_environment'].transform(environment)
    indoor_temp_co2 = models_dict['model_next_state'](torch.tensor(environment).float())
    air_temp_suplly_return = models_dict['model_air_temp_suplly_return'](torch.tensor(environment).float())     
    air_flowrate = models_dict['model_air_flowrate'](torch.tensor(environment).float())
    
    
    next_state = np.concatenate([np.array([[next_outdoor_temp]]), indoor_temp_co2.detach_().numpy(), air_temp_suplly_return.detach_().numpy(), air_flowrate.detach_().numpy()], axis=1)
    
    features_energy = np.concatenate([next_state[:,:1],np.array([action[:2]]),next_state[:,1:3],np.array([action[2:4]]),next_state[:,3:],np.array([action[-1:]])], axis=1)
    scaled_features = scalers_dict['augmented_data_scaler'].transform(features_energy)    
    energy = models_dict['energy_model_augmented_data'].predict(scaled_features, verbose=0)
    # Comfort prediction

    # pmv, ppd = comfort_predictor.pmv_ppd_predictor(next_state[:,1], next_state[:,2])
    # Reward calculation
    reward = - 0.4*energy 
    # reward = - 0.4*energy + 0.6*(abs(3-pmv) + 100-ppd)


    return next_state[0], reward[0,0]