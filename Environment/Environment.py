import numpy as np
from tensorflow import keras

def Environment(action, state, models_dict, scalers_dict):
    """
    action: action to be taken by the agent (array)
    state: current state of the environment (array)
    models_dict: dictionary with all the models
    scalers_dict: dictionary with all the scalers
    """

    # Environment evolution
    # fazer prediction of next state com MLPs

    # Energy prediction
    # array_of_features_energy = np.array([next_state, action]).reshape(1, -1)
    scaled_features = scalers_dict['augmented_data_scaler'].transform(array_of_features_energy)    
    energy = models_dict['energy_model_augmented_data'].predict(scaled_features)

    # Comfort prediction


    # Reward calculation
    

    return next_state, reward, done, info