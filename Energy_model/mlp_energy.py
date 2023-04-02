from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




# def normalize_data(xtrain, xtest):
    
#     scaler = preprocessing.StandardScaler()
#     scaler.fit(xtrain)
#     x_n_train = scaler.transform(xtrain)
#     x_n_test = scaler.transform(xtest)

#     return x_n_test, x_n_train

# def hyperparameters_tunning(x_training, y_training):
     
#     parameters={ 'hidden_layer_sizes':[(46,),(42), (40,), (38,),(50,)],  'alpha':[  0.01, 0.001, 0.0005], 'learning_rate_init':[ 0.01,0.001, 0.0001]}
#     model = MLPRegressor()
#     gs_cv1 = GridSearchCV(model , parameters, cv= 5,refit="score")
#     gs_cv1.fit(x_training,y_training)
#     print(gs_cv1.best_params_)
    
#     return

# def prediction_multiclass(xtrain, ytrain):

#     # model = MLPRegressor(alpha=0.0005, hidden_layer_sizes=(40,),activation = 'relu', learning_rate_init=0.005)
#     model = MLPRegressor(alpha=0.0005, hidden_layer_sizes=(50,),activation = 'relu', learning_rate_init=0.01)
#     model.fit(xtrain, ytrain)
#     mse_train = model.score(xtrain, ytrain)
#     print(mse_train)

#     return model

# def predict_regressor(X, Y, model):

#     y_pred = model.predict(X)
#     mse = mean_squared_error(Y, y_pred)

#     mae = mean_absolute_error(Y, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print('MSE: ', mse , 'MAE: ', mae , 'R2: ', r2)

#     return

# ######################################  MAIN ##########################################

# df = pd.read_csv('dataset_building.csv')
# # print(df.columns)
# df.drop(columns=['date'], axis=0, inplace=True)

# ####################################### SPLIT DATA   ###########################################

# y = df['energy_hvac'].copy()
# x = df.drop(columns=['energy_hvac'], axis= 1 , inplace=False)

# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15,random_state=42)

# #######################################  NORMALIZE DATA   #####################################
# # print(y_train)
# x_norm_test, x_norm_train = normalize_data(x_train, x_test)
# # scaler = preprocessing.StandardScaler()
# # y_train_norm = scaler.fit_transform(y_train.reshape(-1, 1))


# #######################################  Hyperparameters tunning #################################

# # hyperparameters_tunning(x_norm_train,y_train)

# #######################################  PREDICTION RESULTS  ######################################

# model_regressor= prediction_multiclass(x_norm_train, y_train)

# predict_regressor(x_norm_test, y_test, model_regressor)


df1 = pd.read_csv('dataset_building.csv')
# data = df1['supplyfan_speed'].where( df1['supplyfan_speed']>= 0)
data = (df1['supplyfan_speed'] < 0 ).sum()
data = (df1['returnfan_speed'] < 0 ).sum()
data = (df1['zone_temp_heating'] > 24 ).sum()
# data = (df1['zone_temp_cooling'] > 30 ).sum()
df1 = df1.drop(df1[df1['supplyfan_speed'] < 0].index)
df1 = df1.drop(df1[df1['zone_temp_heating'] > 24].index)
df1 = df1.drop(df1[df1['zone_temp_cooling'] > 32].index)
print(df1.shape)

print(df1.min())
print(df1.max())

# def plot_histograms(df):
#     num_columns = df.shape[1]
#     fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, 5 * num_columns))

#     for idx, column in enumerate(df.columns):
#         df[column].hist(ax=axes[idx], bins=150, density=True)
#         axes[idx].set_title(f"Histogram of {column}")

#     plt.tight_layout()
#     plt.show()
# plot_histograms(df1)

print(df1.describe())
df1.to_csv("dataset_building.csv", index=False)  
df2 = pd.read_csv('dataset_building.csv')

print(df2.max())


