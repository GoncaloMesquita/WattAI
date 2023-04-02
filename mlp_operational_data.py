import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
import itertools
import torch
import torch.nn as nn
import torch.optim as optim


def normalize_data(xtrain, xtest):
    
    scaler = preprocessing.StandardScaler()
    x_n_train = scaler.fit_transform(xtrain)
    x_n_test = scaler.transform(xtest)

    return x_n_test, x_n_train

def r2_score(y_true, y_pred1):

    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean)**2)
    ss_res = torch.sum((y_true - y_pred1)**2)
    r2 = 1 - ss_res / ss_tot
    return r2

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



df = pd.read_csv('dataset_building.csv')
df_small = df.loc[:,['Outdoor_temp','zone_temp_cooling', 'zone_temp_heating','indoor_temp_interior','co2', 'fan_speed', 'supplyfan_speed']]


x, y = df_small.drop(['indoor_temp_interior'], inplace=False, axis=1) , df_small.loc[:,['indoor_temp_interior', 'supplyfan_speed']]

x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.15, random_state=42)

x_test_norm, x_norm_train = normalize_data(x_train, x_test)


############################################ MLP ############################################################

input_size = 6
hidden_size = 30
output_size = 2
learning_rate = 0.01
num_epochs = 300

model = MLP( input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range (num_epochs):

    optimizer.zero_grad()                                                   # Set the gradients to None
    outputs = model(torch.tensor(x_norm_train).float())                     # Train the model
    loss = criterion(outputs, torch.tensor(y_train.values).float())         # Calculate the loss 
    loss.backward()                                                         # Do the backpropagation 
    optimizer.step()                                                        # Update the weights 
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}, R2: {:.4f}'.format(epoch+1, num_epochs, loss.item(), r2_score(torch.tensor(y_train.values).float() ,outputs)))

with torch.no_grad():

    model.eval()
    y_pred = model(torch.tensor(x_test_norm).float()) 
    print(' Testing -> Loss: {:.4f}, R2: {:.4f}'.format( criterion(y_pred, torch.tensor(y_test.values).float()).item() , r2_score(torch.tensor(y_test.values).float() ,y_pred)))





