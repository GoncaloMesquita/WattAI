import numpy as np
from tensorflow import keras
import pandas as pd
import torch

def Environment(action, state, models_dict, scalers_dict):
    """
    action: action to be taken by the agent (array)
    state: current state of the environment (array)
    models_dict: dictionary with all the models
    scalers_dict: dictionary with all the scalers
    """

    # Environment evolution
    df_state = np.concatenate((action, state ), axis=1)  
    # fazer prediction of next state com MLPs

    indoor_temp_co2 = models_dict['model_next_state'](torch.tensor(df_state).float())
    air_temp_suplly_return = models_dict['model_air_temp_suplly_return'](torch.tensor(df_state).float())      
    air_flowrate = models_dict['model_air_flowrate'](torch.tensor(df_state).float())

    # next_state = np.array([outdoor_next_state, indoor_temp_co2])
    
    # features energy :  action + current state space + operational data 
    
    features_energy = np.concatenate((action, indoor_temp_co2.numpy(), air_temp_suplly_return.numpy(), air_flowrate.numpy()), axis=1).reshape(1, -1)
    # Energy prediction
    # array_of_features_energy = np.array([next_state, action]).reshape(1, -1)
    scaled_features = scalers_dict['augmented_data_scaler'].transform(features_energy)    
    energy = models_dict['energy_model_augmented_data'].predict(scaled_features)

    # Comfort prediction


    # Reward calculation
    

    return next_state, reward, done, info