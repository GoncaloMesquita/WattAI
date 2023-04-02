"""
Generates synthetic data via the noise MLP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
import pickle
import sys
sys.path.append('Feature_engineering_data_visualization')
from data_visualization import plot_histograms





# read the data
data = pd.read_csv('Noise_model/data_noise.csv')

noise_samples = pd.read_csv('Kde_models/kde_noise_independent_sampling.csv')
# noise_samples = pd.read_csv('Kde_models/kde_noise_dependent_sampling.csv')

plot_histograms(noise_samples)

plot_histograms(data)

plt.show()

noise_model = keras.models.load_model('Noise_model/noise_model.h5')
with open('Noise_model/noise_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

X = noise_samples.iloc[:, range(1,12)]
y = noise_samples.iloc[:, 0]

X = scaler.transform(X)
y_pred = noise_model.predict(X)
# noise_samples['energy_hvac'] = y_pred
# noise_samples.to_csv('Kde_models/kde_noise_independent_sampling.csv', index=False)

# Calculate performance metrics
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

