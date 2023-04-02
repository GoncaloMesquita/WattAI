
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
from data_visualization import plot_histograms

# Generate a sample dataframe
# df_x = pd.DataFrame({'x': np.random.normal(0, 1, size=10000)})
# print(df_x.shape)
# df = pd.read_csv('data_noise.csv')
# air_flow = df.iloc[:, 10].to_numpy()

# Generate a sample dataframe with multiple features
# df = pd.DataFrame({
#     'x1': np.random.normal(0, 1, size=1000),
#     'x2': np.random.normal(2, 2, size=1000)
# })

# sample_data = np.random.rand(100, 2)

# Fit a Kernel Density Estimator to the data
# kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(sample_data)
# Estimate the density using kernel density estimation
# bandwidths = 1.06 * np.std(df, axis=0) * len(df) ** (-1/5) # Silverman's rule of thumb
# kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
# kde.fit(df.to_numpy())
cov = np.array([[1, 0.5], [0.5, 1]])
X = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.2], [0.9, 1]], size=10000)
df_x = pd.DataFrame({'x1': X[:,0], 'x2': X[:,1]})
plot_histograms(df_x)

# Compute the bandwidth using the modified Silverman rule
bw = np.power(np.prod(np.diag(cov)), 1/(2*X.shape[1]+4))

# Set a non-zero bandwidth value for each dimension
bandwidth = bw.mean()

# Create a kernel density estimator with the modified Silverman rule
kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')

# Fit the estimator to the data
kde.fit(X)

# Extract samples from the estimated density
num_samples = 10000
samples = kde.sample(num_samples)
samples = pd.DataFrame(samples, columns=['x1', 'x2'])
plot_histograms(samples)
plt.show()
# # Create a range of values to plot the estimated density
# x_min = df['x'].min()
# x_max = df['x'].max()
# x_range = np.linspace(x_min, x_max, 10000).reshape(-1, 1)

# Evaluate the estimated density at each point in the range
# log_dens = kde.score_samples(x_range)

num_samples = 35900
samples = kde.sample(num_samples)
# Plot the estimated density
plt.hist(air_flow[:,None], density=True, alpha=0.5)
plt.hist(samples, density=True, alpha=0.5)
# plt.plot(x_range, np.exp(log_dens), label='Kernel Density Estimate')
plt.legend()
plt.show()



