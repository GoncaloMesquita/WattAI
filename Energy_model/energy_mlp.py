"""
MLP to predict energy consumption based on input features.

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
import pickle

# read the data
# data = pd.read_csv('dataset_building.csv')
# train_data = pd.read_csv('Synthetic_data/clean_synthetic_data.csv')
train_data = pd.read_csv('Training_data.csv')
# validation_data = pd.read_csv('Training_data.csv')
test_data = pd.read_csv('Hold_out_data.csv')
# train_data = pd.concat([train_data, validation_data], axis=0)

X_train = train_data.iloc[:, range(1,12)]
y_train = train_data.iloc[:, 0]
X_test = test_data.iloc[:, range(1,12)]
y_test = test_data.iloc[:, 0]
# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# df = pd.concat([y_train, X_train], axis=1)
# df.to_csv('Training_data.csv', index=False)

# df = pd.concat([y_test, X_test], axis=1)
# df.to_csv('Hold_out_data.csv', index=False)
# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the MLP model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_train[:, None].shape[1]))


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
my_callbacks = [keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=20, restore_best_weights=True)]
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks= my_callbacks)

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Save the trained model and scaler
# model.save('Energy_model/energy_model_normal_data.h5')
# with open('Energy_model/normal_data_scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)