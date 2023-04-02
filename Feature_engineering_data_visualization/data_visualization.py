import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_histograms(df):
    num_columns = df.shape[1]
    fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5 * num_columns))

    for idx, column in enumerate(df.columns):
        df[column].hist(ax=axes[idx], bins=150, density=True)
        axes[idx].set_title(f"Histogram of {column}")

    plt.tight_layout()
    
    # plt.show()

def plot_corr(df, size=10):
    corr_matrix = df.corr()
    plt.figure(figsize=(size, size))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title('Correlation Matrix')
    plt.show()
# read the data
# data = pd.read_csv('dataset_building_updated.csv')

# # data.iloc[:,[6,7,11]] = data.iloc[:,[6,7,11]].replace(0, 1)
# # data.to_csv("dataset_building_updated.csv", index=False)

# df= data.diff(periods=49).dropna()
# df.iloc[:,[6,7,11]] = data.iloc[:,[6,7,11]].pct_change(periods=49).dropna()
# # plot_histograms(df)
# # data_aux = data.iloc[:,[6,7,11]]
# data_aux = df.iloc[:,[2,3]]
# # data_aux = df.iloc[:,[6,7,11]]
# plot_histograms(data_aux)
# df.to_csv("data_noise.csv", index=False)