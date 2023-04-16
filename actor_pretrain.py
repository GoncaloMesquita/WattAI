import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append('RL_pretraining')


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, min_action, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor', chkpt_dir='RL_pretraining/actor_pre'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        # self.fc3_dims = fc3_dim
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.min_action = min_action
        self.action_range = self.max_action - self.min_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.sigma.weight)


        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.device = T.device('cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.leaky_relu(prob)
        prob = self.fc2(prob)
        prob = F.leaky_relu(prob)
        # prob = self.fc3(prob)
        # prob = F.relu(prob)
        
        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=T.tensor([self.reparam_noise,self.reparam_noise,self.reparam_noise,self.reparam_noise,self.reparam_noise
                                             ]).to(self.device), max=T.tensor([2, 2, 30, 30, 44]).to(self.device))
        # sigma = T.clamp(sigma, min=self.reparam_noise, max=30)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        aux = T.tanh(actions)
        action = T.tensor(self.min_action).to(self.device) + 0.5 * (T.tensor(self.action_range).to(self.device) * (aux + 1))
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-aux.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



def pretrain_actor(actor_net, data, scaler, epochs=100, batch_size=32):
    # optimizer = optim.Adam(actor_net.parameters(), lr=0.001)

    n_samples = len(data)
    n_batches = int(n_samples / batch_size)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            state_batches, action_batches = [], []
            batch_start_idx = batch_idx * batch_size
            batch_end_idx = (batch_idx + 1) * batch_size

            # for i in range(batch_start_idx, batch_end_idx):
            #     state_batch, action_batch = np.array([data.iloc[i,[1,4,5,8,9,10]].values]), np.array([data.iloc[i,[2,3,6,7,11]].values])
            #     state_batches.append(scaler['state_scalers'].transform(state_batch)[0])
            #     action_batches.append(scaler['actions_scalers'].transform(action_batch)[0])
            state_batch, action_batch = np.array(data.iloc[range(batch_start_idx, batch_end_idx),[1,4,5,8,9,10]].values), np.array(data.iloc[range(batch_start_idx, batch_end_idx),[2,3,6,7,11]].values)
            state_batches.append(scaler['state_scalers'].transform(state_batch)[0])
            action_batches.append(action_batch[0])
            # action_batches.append(scaler['actions_scalers'].transform(action_batch)[0])

            state_batches = T.FloatTensor(state_batches).to(actor_net.device)
            action_batches = T.FloatTensor(action_batches).to(actor_net.device)

            predicted_actions, _ = actor_net.sample_normal(state_batches, reparameterize=False)

            # predicted_actions = scaler['actions_scalers'].transform(predicted_actions.cpu().detach().numpy())
            # predicted_actions = T.FloatTensor(predicted_actions).to(actor_net.device)
            action_batches.requires_grad = True
            predicted_actions.requires_grad = True

            loss = F.mse_loss(predicted_actions.double(), action_batches.double()).float()
            epoch_loss += loss.item()

           


            actor_net.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            actor_net.optimizer.step()

        epoch_loss /= n_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    actor_net.save_checkpoint()

if __name__ == '__main__':
    data_set_environment = pd.read_csv('Environment/data_set_environment.csv')
    from Gym_env_pre_train import Real_Environment
    from env_pre_train import Environment_real_data

    env = Real_Environment(Environment_real_data, 6, 5, data_set_environment)
    actor = ActorNetwork(alpha=0.0008,fc1_dims=64 ,fc2_dims= 64, input_dims=env.observation_space.shape, n_actions=env.action_space.shape[0], max_action=env.action_space.high, min_action= env.action_space.low, chkpt_dir='RL_pretraining/actor_pre')
    actor.load_checkpoint()

    scalers = ['Energy_model/augmented_data_scaler.pkl', 'Environment/scaler_environment.pkl', 'Scalers/state_scalers.pkl', 'Scalers/actions_scalers.pkl']
    scalers_dict = dict()
    for sca in scalers:
        scaler = sca.split('/')[-1].split('.')[0]
        with open(sca, 'rb') as f:
            scalers_dict[scaler] = pickle.load(f)

    pretrain_actor(actor, data_set_environment, scalers_dict, epochs=500, batch_size=35486)

