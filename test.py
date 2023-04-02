
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from scipy.stats import norm
from sklearn.neighbors import KernelDensity


# read the data
data = pd.read_csv('dataset_building.csv')
# correlation = data.corr()
# print(np.linalg.norm(correlation.iloc[:,0]))
# # data = pd.read_csv('noise_model/data_noise.csv')
# norms = []
# for i in range(1,70):
#     data_aux = data.diff(periods=i).dropna()
#     correlation = data_aux.corr()
#     norms.append(np.linalg.norm(correlation.iloc[:,0]))
# print(np.argmax(norms))
# print(np.max(norms))
data_aux = data.diff(periods=50).dropna()
data_aux.iloc[:,[6,7,11]] = data.iloc[:,[6,7,11]].pct_change(periods=50).dropna()
# correlation = data_aux.corr()
# print(np.linalg.norm(correlation.iloc[:,0]))
data_aux.to_csv('Noise_model/data_noise.csv', index=False)

# data = data.diff(periods=49).dropna()



