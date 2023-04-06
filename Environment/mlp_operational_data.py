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
# from torcheval.metrics import R2Score
from sklearn.metrics import r2_score


def normalize_data(xtrain, xtest):
    
    scaler = preprocessing.StandardScaler()
    x_n_train = scaler.fit_transform(xtrain)
    x_n_test = scaler.transform(xtest)

    return x_n_test, x_n_train

def r2_score1(y_true, y_pred1):

    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean)**2)
    ss_res = torch.sum((y_true - y_pred1)**2)
    r2 = 1 - ss_res / ss_tot
    return r2

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size1,hidden_size2 ,hidden_size3,dropout_prob, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        # self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        # self.relu = nn.ReLU()
        # self.dropout3 = nn.Dropout(dropout_prob)
        # self.fc4 = nn.Linear(hidden_size3, output_size)
        self.fc4 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)    
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.dropout3(x)
        x = self.fc4(x)

        return x



############################################ MLP ############################################################

def run_model(x_norm_train,y_train, x_norm_test, y_test ):
    input_size =11
    hidden_size1 = 60
    hidden_size2 = 40
    hidden_size3 = 10
    dropout_prob = 0
    output_size = 1
    learning_rate = 0.15
    num_epochs = 500

    model = MLP( input_size, hidden_size1,hidden_size2,hidden_size3,dropout_prob, output_size)
    # model = MLP( input_size, hidden_size1,dropout_prob, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range (num_epochs):

        optimizer.zero_grad()                                                   # Set the gradients to None
        outputs = model(torch.tensor(x_norm_train).float())                     # Train the model
        loss = criterion(outputs, torch.tensor(y_train.values).float())         # Calculate the loss 
        loss.backward()                                                         # Do the backpropagation 
        optimizer.step()                                                        # Update the weights 
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, R2: {:.4f}'.format(epoch+1, num_epochs, loss.item(), r2_score1(torch.tensor(y_train.values).float() ,outputs)))

    with torch.no_grad():

        model.eval()
        y_pred = model(torch.tensor(x_norm_test).float()) 
        print(' Testing -> Loss: {:.4f}, R2: {:.4f}'.format( criterion(y_pred, torch.tensor(y_test.values).float()).item() , r2_score1(torch.tensor(y_test.values).float() ,y_pred)))
    

    print('R2:', r2_score(y_test.to_numpy(), y_pred.numpy() ))
    return y_pred



##################################################### Main #####################################################

train_data = pd.read_csv('Synthetic_data/clean_synthetic_data.csv')
validation_data = pd.read_csv('Training_data.csv')
df = pd.concat([train_data, validation_data], axis=0, ignore_index=True)
# x, y = df_small.drop(['indoor_temp_interior'], inplace=False, axis=1) , df_small.loc[:,['indoor_temp_interior', 'co2']]

y_ns= pd.DataFrame(df.loc[:,['indoor_temp_interior', 'co2','supply_air_temp', 'return_air_temp', 'filtered_air_flow_rate']], columns=['ns_indoor_temp_interior', 'ns_co2','ns_supply_air_temp', 'ns_return_air_temp', 'filtered_air_flow_rate'])
y_ns = df.loc[:,['indoor_temp_interior', 'co2','supply_air_temp', 'return_air_temp', 'filtered_air_flow_rate']]
df = df.iloc[:-1]
x_ns = y_ns.iloc[1:]
x_ns = x_ns.reset_index(drop=True)
y_ns = y_ns.iloc[:-1]

# y_ns.drop(['co2'], inplace=True, axis=1)
# print(y_ns)

x_action = df.loc[:,['zone_temp_cooling', 'zone_temp_heating', 'supplyfan_speed', 'returnfan_speed', 'outdoor_air_damper_position', 'Outdoor_temp']]
x_action_state = pd.concat([x_action, x_ns], axis=1)   



df_ns= pd.concat([df, x_ns ], axis=1, ignore_index=True)


df_ns.to_csv('Environment/data_set_environment.csv')

x_train, x_test, Y_train, Y_test = train_test_split(x_action_state , y_ns , test_size=0.15, random_state=42)

X_norm_test, X_norm_train = normalize_data(x_train, x_test)

y_ns_pred = run_model(X_norm_train,Y_train, X_norm_test, Y_test)

print(y_ns_pred.numpy(), Y_test.to_numpy())


