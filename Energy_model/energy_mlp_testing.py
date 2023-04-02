"""
Testing MLP to predict energy consumption based on input features.

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import pickle

def test_model(model, X_test, y_test):
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # print(f"Mean Squared Error: {mse}")
    # print(f"Mean Absolute Error: {mae}")
    # print(f"R-squared: {r2}")

    return r2

# read the data
test_data = pd.read_csv('Hold_out_data.csv')
X_test = test_data.iloc[:, range(1,12)]
y_test = test_data.iloc[:, 0]

models = ['Energy_model/energy_model_normal_data.h5', 'Energy_model/energy_model_syn_data.h5', 'Energy_model/energy_model_augmented_data.h5']
scalers = ['Energy_model/normal_data_scaler.pkl', 'Energy_model/syn_data_scaler.pkl', 'Energy_model/augmented_data_scaler.pkl']
R_2 = dict()
for mod,sca in zip(models,scalers):
    energy_model = keras.models.load_model(mod)
    with open(sca, 'rb') as f:
        scaler = pickle.load(f)
    X_test_aux = scaler.transform(X_test)
    filename = mod.split('/')[-1].split('.')[0]
    R_2[filename] = test_model(energy_model, X_test_aux, y_test)

print('R^2: ', R_2)



