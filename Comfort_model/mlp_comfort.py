"""

The goal of this script is to use an MLP to predict the comfort of a person in a room 
based on the temperature and CO2 levels.

Implemented by Diogo Araújo.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import pickle

#from torchmetrics.functional import r2_score

import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import argparse
from pathlib import Path

import os

from typing import Dict, List, Tuple
from tqdm.auto import tqdm

import comfort_predictor as cp

# import EarlyStopping
#from pytorccondhtools import EarlyStopping

NUM_WORKERS = os.cpu_count()

#from pytorch.ppg_feature_dataset import FeatureDataset

def get_args_parser():
   
    parser = argparse.ArgumentParser('MLP operational predictor', add_help=False)
    
    ## Add arguments here
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--train_file', default='Synthetic_data/clean_synthetic_data.csv', help='path to input file')
    parser.add_argument('--test_file', default='Hold_out_data.csv', help='path to input file')
    parser.add_argument('--validation_file', default='Training_data.csv', help='path to input file')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    
    # Training parameters
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    
    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.005)
        
    # Optimizer
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd')   
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    
    # Loss function 
    parser.add_argument('--loss', choices=['mse', 'mae'], default='mse')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    
    # Architecture parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64,16])
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--activation', choices=['relu', 'tanh'], default='relu')

 
    return parser

def Predict_Comfort(input_file):
    
    df = pd.read_csv(input_file)
    pmv, ppd = cp.pmv_ppd_predictor(np.array(df['indoor_temp_interior'].values), 
                                    np.array(df['co2'].values))
    
    df['pmv'] = pmv
    df['ppd'] = ppd
    
    df = df[['co2', 'indoor_temp_interior', 'pmv', 'ppd']]
        
    return df

def Process_Data(train_data, test_data, args = None):
    
    # Split the train and test datasets into X (features) and y (target)
    # Targes : pmv and ppd
    x_train = train_data.iloc[:,0:-2].values
    y_train = train_data.iloc[:,-2:].values
    x_test = test_data.iloc[:,0:-2].values
    y_test = test_data.iloc[:,-2:].values
    
    # Apply feature scaling to the train and test datasets separately
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    # Save the scaler
    pickle.dump(sc, open(args.output_dir + 'mlp_comfort_scaler.pkl','wb'))

    return x_train, y_train, x_test, y_test

def Process_Data_Env(train_data, test_data, args = None):
    
    train_data = train_data.drop(['supply_air_temp',
                                  'return_air_temp',
                                  'filtered_air_flow_rate'], axis=1)
    
    test_data = test_data.drop(['supply_air_temp',
                                'return_air_temp',
                                'filtered_air_flow_rate'], axis=1)
    
    print(train_data.iloc[:,-2:].columns)
    print(test_data.iloc[:, 0:-2].columns)
    # Split the train and test datasets into X (features) and y (target)
    # Targes : pmv and ppd
    x_train = train_data.iloc[:,0:-2].values
    y_train = train_data.iloc[:,-2:].values
    x_test = test_data.iloc[:,0:-2].values
    y_test = test_data.iloc[:,-2:].values
    
    # Apply feature scaling to the train and test datasets separately
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    # Save the scaler
    pickle.dump(sc, open(args.output_dir + 'mlp_comfort_scaler.pkl','wb'))

    return x_train, y_train, x_test, y_test

class FeatureDataset(Dataset):
    
    def __init__(self, x, y):
        
        # Convert the train and test datasets to PyTorch tensors
        self.X = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
 
class MLP(nn.Module):
    
    def __init__(self, n_features, n_outputs, layers,
            activation_type, dropout, **kwargs):
        """
        n_features (int)
        n_outputs (int)
        layers (list): list of hidden layers sizes
        activation_type (str)
        dropout (float): dropout probability
        """
        super().__init__()
        
        activation_func = {"relu": nn.ReLU(), "tanh": nn.Tanh()}
        
        # The first hidden layer has size (n_features, hidden_size).
        hidden_input_size = n_features

        # (2) We need to create a list that will contain our hidden layers.
        hidden_layers = []

        # (3) Compose our feedfoward network accordingly with the hyperparameters chosen.
        for hidden_size in layers:
                        
            # We will had the hidden layer to our feedfoward network.
            hidden_layers.append(nn.Linear(hidden_input_size, hidden_size))

            # Then we need to define the activation function at the end of each hidden layer.
            hidden_layers.append(activation_func[activation_type])
            
            # Then we need to apply dropout to prevent overfitting.
            hidden_layers.append(nn.Dropout(dropout))
            
            # Only the first hidden layer has dimensions (n_features, hidden_size).
            # All the other hidden layers (2nd and 3rd) will have (hidden_size, hidden_size).
            hidden_input_size = hidden_size
        
        # (4) Add the output layer to our feedfoward network.
        hidden_layers.append(nn.Linear(hidden_size,n_outputs))
        
        # (5) Create a Sequence of pytorch models -> Now we have our Feedfoward Network
        self.ff = nn.Sequential(*hidden_layers)
            
        
    def forward(self, x, **kwargs):
        """
        x (batch_size x n_features): a batch of training examples
        """
        return self.ff(x)

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.
    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)
    
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float, float]:
    """Trains a PyTorch model for a single epoch on a regression problem
    
    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss, train mse and train r2 values
    train_loss, train_mse, train_r2 = 0, 0, 0

    # Loop through data loader data batches
    for _, (X, y) in enumerate(dataloader):
        
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate mse and r2 scores
        y_pred_np = y_pred.detach().cpu().numpy().squeeze()
        y_np = y.detach().cpu().numpy().squeeze()
                
        train_mse += ((y_pred_np - y_np) ** 2).mean()
        train_r2 += r2_score(y_np, y_pred_np)

    # Adjust metrics to get average loss, mse and r2 score per batch 
    train_loss = train_loss / len(dataloader)
    train_mse = train_mse / len(dataloader)
    train_r2 = train_r2 / len(dataloader)
    
    return train_loss, train_mse, train_r2

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss, mse and r2_score values
    test_loss, test_mse, test_r2 = 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        
        # Loop through DataLoader batches
        for X, y in dataloader:
            
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

            # Calculate and accumulate mse
            test_mse += ((test_pred - y)**2).mean().item()

            # Calculate and accumulate r2 score
            test_r2 += r2_score(y.cpu().detach().numpy(), test_pred.cpu().detach().numpy())


    # Adjust metrics to get average loss, mse and r2 score per batch 
    test_loss = test_loss / len(dataloader)
    test_mse = test_mse / len(dataloader)
    test_r2 = test_r2 / len(dataloader)

    return test_loss, test_mse, test_r2

def engine(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          args = None) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_mse": [],
               "train_r2": [],
               "test_loss": [],
               "test_mse": [],
               "test_r2": []
    }
    
    # Make sure model on target device
    model.to(device)
    
    # Early stopping
    early_stopping = EarlyStopping(tolerance=5, min_delta=10)

    min_test_r2 = 0

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
      
        train_loss, train_mse, train_r2 = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        
        test_loss, test_mse, test_r2 = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_mse: {train_mse:.4f} | "
          f"train_r2: {train_r2:.4f} |"
          f"test_loss: {test_loss:.4f} | "
          f"test_mse: {test_mse:.4f} |"
          f"test_r2: {test_r2:.4f} "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_mse"].append(train_mse)
        results["train_r2"].append(train_r2)
        results["test_loss"].append(test_loss)
        results["test_mse"].append(test_mse)
        results["test_r2"].append(test_r2)
        
        # Save model if test r2 score is better than previous best
        if test_r2 > min_test_r2:
            min_test_r2 = test_r2
            """ save_model(model=model, 
                       target_dir= args.output_dir,
                       model_name="best_mlp_comfort.pth") """
            torch.save(model , args.output_dir + "best_mlp_comfort.pt")
        
        # early stopping
        early_stopping(train_loss, test_loss)
        if early_stopping.early_stop:
            print(f"====== Early stopping at epoch {epoch+1} ======")
            break

    # Return the filled results at the end of the epochs
    return results

def plot_loss_curves(results, output_dir):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "test_loss": [...],
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    epochs = range(len(results["train_loss"]))

    # Plot loss
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss.png"))

def plot_r2_curves(results, output_dir):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "test_loss": [...],
    """
    loss = results["train_r2"]
    test_loss = results["test_r2"]

    epochs = range(len(results["train_r2"]))

    # Plot loss
    plt.close()
    plt.cla()
    plt.plot(epochs, loss, label="train_r2")
    plt.plot(epochs, test_loss, label="test_r2")
    plt.title("r2")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "r2.png"))
    
def main(args):
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    
    ############# Predict Comfort (targets) ############
        
    df_building = Predict_Comfort(args.validation_file)
    df_synthetic = Predict_Comfort(args.train_file)
    df_test = Predict_Comfort(args.test_file)
        
    # Concatenate the two dataframes
    df_train = pd.concat([df_synthetic, df_building], axis=0)
    
    """ df_train = pd.read_csv(args.train_file)
    df_test = pd.read_csv(args.test_file)
    
    print(df_train.describe()) """
                    
    # Process the data
    x_train, y_train, x_test, y_test = Process_Data(df_train, df_test, args=args)
    #x_train, y_train, x_test, y_test = Process_Data_Env(df_train, df_test, args=args)
    
        
    # Create the dataset
    train_dataset = FeatureDataset(x_train, y_train)
    test_dataset = FeatureDataset(x_test, y_test)
    
    # Split the dataset into training and test sets
    """ x_train, x_test, y_train, y_test = train_test_split(feature_dataset.X , feature_dataset.y , test_size=0.10, 
                                                        random_state=42) """

    # Create the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True,
                                                   num_workers=NUM_WORKERS,     
                                                   pin_memory=True)
    
    test_dataset = torch.utils.data.DataLoader(test_dataset, 
                                               batch_size=args.batch_size, 
                                               shuffle=False, 
                                               num_workers=NUM_WORKERS, 
                                               pin_memory=True)
    
    # Define the model  
    model = MLP(args.input_size, args.output_size, args.hidden_layers, 
                args.activation, args.dropout).to(device)
    
    print("----------- Model Architecture -----------")
    print(model)

    # Define the loss function
    criterion = nn.MSELoss() if args.loss == 'mse' else nn.L1Loss()
    
    # Define the optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)  
    else: 
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
          
    # Train and test the model
    results = engine(model=model,
                     train_dataloader=train_dataloader,
                     test_dataloader=test_dataset,
                     optimizer=optimizer,
                     loss_fn=criterion,
                     epochs=args.epochs,
                     device=device,
                     args=args)    
    
    # Plot the loss curves
    plot_loss_curves(results, args.output_dir)
    plot_r2_curves(results, args.output_dir)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MLP operational predictor', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)