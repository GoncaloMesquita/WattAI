from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from KDEpy import FFTKDE

def silverman_bandwidth(data):
    n, d = data.shape
    iqr = np.subtract(*np.percentile(data, [75, 25], axis=0))
    silverman = np.power(n, -1 / (d + 4))
    return 0.9 * np.minimum(np.std(data, axis=0), iqr / 1.34) * silverman

def kernel_estimator(df, col=True):
    if col:
        for column in df.columns:
            bandwidth = 1.06 * df[column].std() * len(df[column]) ** (-1/5) # Silverman's rule of thumb
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            # kde = KernelDensity(bandwidth='silverman', kernel='gaussian')
            aux = df[column].to_numpy()
            kde.fit(aux[:,None])
            file_name = os.path.join(directory, file_name_template.format(column))
            with open(file_name, 'wb') as f:
                pickle.dump(kde, f)
    else:
        # bandwidth = silverman_bandwidth(df)
        # print(bandwidth)
        # cov = np.cov(df)
        # bw = np.power(np.prod(np.diag(cov)), 1/(2*df.shape[1]+4))
        # bandwidth = bw.mean()
        kde = KernelDensity(bandwidth=0.01, kernel='gaussian')
        kde.fit(df.values)
        file_name = os.path.join(directory, file_name_template.format('noise'))
        with open(file_name, 'wb') as f:
                    pickle.dump(kde, f)
directory = 'Kde_models'
file_name_template = 'kde_{}.pkl'
df = pd.read_csv('noise_model/data_noise.csv')
df_samples = pd.DataFrame(columns=df.columns, index=range(35827))
kernel_estimator(df, col=False)
num_samples = 35827
flag = False
if flag:
    for column in df.columns:
        file_name = os.path.join(directory, file_name_template.format(column))
        with open(file_name, 'rb') as f:
            kde = pickle.load(f)
        samples = kde.sample(num_samples)
        df_samples[column] = samples
        # Plot the estimated density    
        plt.hist(df[column], density=True, bins=100, alpha=0.5, label='original')
        plt.hist(samples, density=True, bins=100, alpha=0.5,label='estimated')
        plt.title(column)
        plt.legend()
        # plt.show()
    # df_samples.to_csv('Kde_models/kde_noise_independent_sampling.csv', index=False) 
else:
    file_name = os.path.join(directory, file_name_template.format('noise'))
    with open(file_name, 'rb') as f:
        kde = pickle.load(f)
    samples = kde.sample(num_samples)
    df_samples.iloc[:,:] = samples
    # df_samples.to_csv('Kde_models/kde_noise_dependent_sampling.csv', index=False)