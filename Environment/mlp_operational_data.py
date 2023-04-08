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
from torchmetrics.functional import r2_score
import pickle
# from torcheval.metrics import R2Score
# from sklearn.metrics import r2_score


def normalize_data(xtrain, xtest):
    
    scaler = preprocessing.StandardScaler()
    x_n_train = scaler.fit_transform(xtrain)
    x_n_test = scaler.transform(xtest)
    with open('Environment/scaler_environment.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return x_n_test, x_n_train

def r2_score1(y_true, y_pred1):

    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean)**2)
    ss_res = torch.sum((y_true - y_pred1)**2)
    r2 = 1 - ss_res / ss_tot
    return r2

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size1,hidden_size2 ,dropout_prob, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(hidden_size2 , output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)    
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc4(x)

        return x



############################################ MLP ############################################################

def run_model_enviorment(x_norm_train,y_train, x_norm_test, y_test, size_output ):
    input_size =11
    hidden_size1 = 26
    hidden_size2 = 26
    
    dropout_prob = 0
    output_size = size_output
    learning_rate = 0.15
    num_epochs = 500

    model = MLP( input_size, hidden_size1,hidden_size2, dropout_prob, output_size)
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
            mean_r2score_train = (r2_score(outputs[:,0],torch.tensor(y_train.iloc[:,0].values).float()) + r2_score(outputs[:,1],torch.tensor(y_train.iloc[:,1].values).float()) )/ y_train.shape[1]
            print('Epoch [{}/{}], Loss: {:.4f}, R2: {:.4f}'.format(epoch+1, num_epochs, loss.item(), mean_r2score_train))

    with torch.no_grad():

        model.eval()
        y_pred = model(torch.tensor(x_norm_test).float()) 
        mean_r2score_test = (r2_score(y_pred[:,0],torch.tensor(y_test.iloc[:,0].values).float()) + r2_score(y_pred[:,1],torch.tensor(y_test.iloc[:,1].values).float()) )/ y_test.shape[1]
        print(' Testing -> Loss: {:.4f}, R2: {:.4f}'.format( criterion(y_pred, torch.tensor(y_test.values).float()).item() ,mean_r2score_test))
    
    return y_pred, model

def run_model_air_temp(x_norm_train,y_train, x_norm_test, y_test, size_output ):
    input_size =11
    hidden_size1 = 16
    hidden_size2 = 16
    dropout_prob = 0
    output_size = size_output
    learning_rate = 0.15
    num_epochs = 500

    model = MLP( input_size, hidden_size1,hidden_size2,dropout_prob, output_size)
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
            mean_r2score_train = (r2_score(outputs[:,0],torch.tensor(y_train.iloc[:,0].values).float()) + r2_score(outputs[:,1],torch.tensor(y_train.iloc[:,1].values).float()) )/ y_train.shape[1]
            print('Epoch [{}/{}], Loss: {:.4f}, R2: {:.4f}'.format(epoch+1, num_epochs, loss.item(), mean_r2score_train))

    with torch.no_grad():

        model.eval()
        y_pred = model(torch.tensor(x_norm_test).float())
        mean_r2score_test = (r2_score(y_pred[:,0],torch.tensor(y_test.iloc[:,0].values).float()) + r2_score(y_pred[:,1],torch.tensor(y_test.iloc[:,1].values).float()) )/ y_test.shape[1]
        print(' Testing -> Loss: {:.4f}, R2: {:.4f}'.format( criterion(y_pred, torch.tensor(y_test.values).float()).item() ,mean_r2score_test))
    
    return y_pred, model

def run_model_air_flowrate(x_norm_train,y_train, x_norm_test, y_test, size_output ):

    input_size = 11
    hidden_size1 = 30
    hidden_size2 = 30
    dropout_prob = 0
    output_size = size_output
    learning_rate = 0.15
    num_epochs = 500

    model = MLP( input_size, hidden_size1,hidden_size2,dropout_prob, output_size)
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
            print('Epoch [{}/{}], Loss: {:.4f}, R2: {:.4f}'.format(epoch+1, num_epochs, loss.item(), r2_score(outputs,torch.tensor(y_train.values).float())))

    with torch.no_grad():

        model.eval()
        y_pred = model(torch.tensor(x_norm_test).float()) 
        print(' Testing -> Loss: {:.4f}, R2: {:.4f}'.format( criterion(y_pred, torch.tensor(y_test.values).float()).item() ,r2_score( y_pred,torch.tensor(y_test.values).float())))
    
    return y_pred, model





##################################################### Main #####################################################

df = pd.read_csv('dataset_building.csv')

y_ns= pd.DataFrame(df.loc[:,['indoor_temp_interior', 'co2','supply_air_temp', 'return_air_temp', 'filtered_air_flow_rate']].values, columns=['ns_indoor_temp_interior', 'ns_co2','ns_supply_air_temp', 'ns_return_air_temp', 'ns_filtered_air_flow_rate'])

df = df.iloc[:-1]
x_ns = y_ns.iloc[1:]
x_ns = x_ns.reset_index(drop=True)
y_ns = y_ns.iloc[:-1]


# y_next_sate = y_ns.drop([ 'ns_supply_air_temp', 'ns_return_air_temp', 'ns_filtered_air_flow_rate'], axis=1)
# y_operational_data = y_ns.drop([ 'ns_indoor_temp_interior', 'ns_co2'], axis=1)
#print(y_ns)
# y_ns.drop(['ns_indoor_temp_interior', 'ns_co2'],inplace=True, axis=1)

print(df['indoor_temp_interior'])
print(x_ns)

x_action = df.loc[:,['zone_temp_cooling', 'zone_temp_heating', 'supplyfan_speed', 'returnfan_speed', 'outdoor_air_damper_position', 'Outdoor_temp']]
x_action_state = pd.concat([x_action, x_ns], axis=1)   


df_ns = pd.concat([df, x_ns ], axis=1, ignore_index=False)
df_ns.to_csv('Environment/data_set_environment.csv', index=False)

x_train, x_test, Y_train, Y_test = train_test_split(x_action_state , df.loc[:,['indoor_temp_interior', 'co2']], test_size=0.15, random_state=42)

Training = pd.concat([x_train, Y_train ], axis=1, ignore_index=False)
Training.to_csv('Environment/training_environment.csv', index=False)

Testing = pd.concat([x_test, Y_test ], axis=1, ignore_index=False)
Testing.to_csv('Environment/testing_environment.csv', index=False)

X_norm_test, X_norm_train = normalize_data(x_train, x_test)

#######################################  Enviorment MODEL   ######################################

y_ns_pred, model_next_state = run_model_enviorment(X_norm_train,Y_train, X_norm_test, Y_test, 2)


#######################################  Air Temp MODEL   #############################################

x_train_at, x_test_at, Y_train_at, Y_test_at = train_test_split(x_action_state , df.loc[:,['supply_air_temp', 'return_air_temp' ]], test_size=0.15, random_state=42)

#X_norm_test_at, X_norm_train_at = normalize_data(x_train_at, x_test_at)

y_operational_data_pred, model_air_temp_suplly_return  = run_model_air_temp(X_norm_train , Y_train_at, X_norm_test, Y_test_at, 2)

#######################################  Air Flow rate model #########################################

x_train_fr, x_test_fr, Y_train_fr, Y_test_fr = train_test_split(x_action_state , df.loc[:,['filtered_air_flow_rate']], test_size=0.15, random_state=42)

#X_norm_test_fr, X_norm_train_fr = normalize_data(x_train_fr, x_test_fr)

y_operational_data_pred, model_air_flowrate  = run_model_air_flowrate(X_norm_train , Y_train_fr, X_norm_test, Y_test_fr, 1)


# Save the models 

torch.save(model_next_state, 'Environment/model_next_state.pth')
torch.save(model_air_temp_suplly_return, 'Environment/model_air_temp_suplly_return.pth')
torch.save(model_air_flowrate, 'Environment/model_air_flowrate.pth')

