"""
MLP to calculate output noise based on input noise.

MLP_1:
input: columns 1,2,3,4,6
output: columns 7,8,9,10,11,12,13,14,15,16

MLP_2:
input: columns 7,8,9,10,11,12,13,14,15,16
output: columns 0
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load your dataset


def plot_corr(df, size=10):
    corr_matrix = df.corr()
    plt.figure(figsize=(size, size))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title('Correlation Matrix')
    plt.show()
    # corr_matrix.to_csv('corr_matrix.csv')


# read the data
data = pd.read_csv('dataset_building.csv')
# drop the first column
data = data.drop(data.columns[0], axis=1) 
# norms=[]
# for i in range(1, 70):
#     df = data.diff(periods=i).dropna()
#     corr_matrix = df.corr()
#     norms.append(np.linalg.norm(corr_matrix.iloc[:, 0]))

# j = np.argmax(norms)
# print(j)
# print(norms[j])
# print(max(norms))
# data = data.diff(periods=49).dropna() # 49 maximizes the correlation between the output and the input


# plot_corr(data.iloc[:, [0,7,8,9,10,11,12,13,14,15,16]], 15)
# plot_corr(data.iloc[:, [0,8,9,13,14]],15)  # features with highest correlation for energy prediction
# plot_corr(data, 15)

# plot_corr(data.iloc[:, [1,4,8,9,13,14]],15) # features with highest correlation for MLP 1

# X = data.iloc[:, [1,4]]
# y = data.iloc[:, [8,9,13,14]]
# X = data.iloc[:, 1:6]
# y = data.iloc[:, 7:16]

X = data.iloc[:, [7,8,9,11,12,13,14,15]]
# X = data.iloc[:, 7:16]
y = data.iloc[:, 0]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the MLP model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(y[:, None].shape[1]))


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=15, restore_best_weights=True)]
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks= my_callbacks)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
# y_pred = y_pred.reshape(-1)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")
