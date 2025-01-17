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
import math
import sys
sys.path.append('Feature_engineering_data_visualization')
from data_visualization import plot_histograms


def generate_data(original_data, noise, n_samples):
    data_len = len(original_data)

    new_data = pd.DataFrame(columns=original_data.columns, index=range(n_samples*data_len))
    # inject noise into new data
    for i in range(len(new_data)):
        noise_sample = noise.sample(1, ignore_index=True)
        new_data.iloc[i, [0,1,2,3,4,5,8,9,10]] = original_data.iloc[math.floor(i/n_samples), [0,1,2,3,4,5,8,9,10]] + noise_sample.iloc[:, [0,1,2,3,4,5,8,9,10]]
        new_data.iloc[i, [6,7,11]] = original_data.iloc[math.floor(i/n_samples), [6,7,11]] * (1+noise_sample.iloc[:, [6,7,11]])
        if i%200 == 0:
            print(i)
    # lembrar de verificar se não há percentagens acima de 100 nem energias negativas and airflow negativo
    return new_data


# data = pd.read_csv('Training_data.csv')
data = pd.read_csv('dataset_building.csv')
# noise = pd.read_csv('Kde_models/kde_noise_dependent_sampling.csv')
# synthetic_data = generate_data(data, noise, 2)
# synthetic_data.to_csv('Synthetic_data/synthetic_data.csv', index=False)


clean_synthetic_data = pd.read_csv('Synthetic_data/clean_synthetic_data.csv')
synthetic_data = pd.read_csv('Synthetic_data/synthetic_data.csv')

# plot_histograms(data)
# plot_histograms(synthetic_data)
# plot_histograms(clean_synthetic_data)
# plt.show()

# df1 = pd.read_csv('Synthetic_data/synthetic_data.csv')
# df1 = df1.drop(df1[df1['supplyfan_speed'] > 100].index)
# df1 = df1.drop(df1[df1['returnfan_speed'] > 100].index)
# df1 = df1.drop(df1[df1['energy_hvac']  < 0].index)
# df1 = df1.drop(df1[df1['zone_temp_heating'] > 31].index)
# df1 = df1.drop(df1[df1['Outdoor_temp'] > 45].index)
# df1 = df1.drop(df1[df1['outdoor_air_damper_position'] > 100].index)
# df1 = df1.drop(df1[df1['filtered_air_flow_rate'] < 0].index)
# df1 = df1.drop(df1[df1['outdoor_air_damper_position'] < 0].index)
# print(df1.shape)
# print(df1.max())
# print(df1.min())