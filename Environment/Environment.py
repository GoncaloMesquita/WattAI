import numpy as np
from tensorflow import keras
import pandas as pd
import torch

def Environment(action, state, models_dict, scalers_dict, next_outdoor_temp):
    """
    action: action to be taken by the agent (array)
    state: current state of the environment (array)
    models_dict: dictionary with all the models
    scalers_dict: dictionary with all the scalers
    """

    # Environment evolution
    environment = np.concatenate((action, state), axis=1)
    environment = scalers_dict['scaler_environment'].transform(environment)
    indoor_temp_co2 = models_dict['model_next_state'](torch.tensor(environment).float())
    air_temp_suplly_return = models_dict['model_air_temp_suplly_return'](torch.tensor(environment).float())      
    air_flowrate = models_dict['model_air_flowrate'](torch.tensor(environment).float())

    next_state = np.array([next_outdoor_temp, indoor_temp_co2, air_temp_suplly_return, air_flowrate])
    
    # Energy prediction
    features_energy = np.array([state[0],action[:2],state[1:3],action[2:4],state[3:],action[-1]])
    scaled_features = scalers_dict['augmented_data_scaler'].transform(features_energy)    
    energy = models_dict['energy_model_augmented_data'].predict(scaled_features)

    # Comfort prediction


    # Reward calculation
    

    return next_state, reward, done, info