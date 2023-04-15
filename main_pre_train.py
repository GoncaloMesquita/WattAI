import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
import torch
import pandas as pd
import sys
sys.path.append('RL_agent')
sys.path.append('RL_pretraining')
sys.path.append('Synthetic_data')
sys.path.append('Comfort_model')
sys.path.append('Environment')
sys.path.append('Energy_model')
from sac_pre_training import Agent
from Gym_env_pre_train import Real_Environment
from env_pre_train import Environment_real_data
from utils import plot_learning_curve

data_set_environment = pd.read_csv('Environment/data_set_environment.csv')
        
# load all models and scalers
scalers = ['Energy_model/augmented_data_scaler.pkl', 'Environment/scaler_environment.pkl', 'Scalers/state_scalers.pkl', 'Scalers/actions_scalers.pkl']
scalers_dict = dict()
for sca in scalers:
    scaler = sca.split('/')[-1].split('.')[0]
    with open(sca, 'rb') as f:
        scalers_dict[scaler] = pickle.load(f)

env = Real_Environment(Environment_real_data, 6, 5, data_set_environment)

agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0], gamma=0.99,alpha = 0.003,beta = 0.003, max_size=256, tau=0.001,
            batch_size=256, reward_scale=5, chkpt_dir='RL_pretraining/sac')
n_games = 1000

filename = 'actor_loss.png'

figure_file = 'Plots/' + filename

# best_score = env.reward_range[0]
# extract the best score from the file best_score.txt
with open('RL_pretraining/best_score.txt', 'r') as f:
    best_loss = float(f.read())



score_history = []
actor_losses = []
# best_loss = 100
load_checkpoint = False

agent.load_models()


for i in range(n_games):
    observation, index = env.reset()
    done = False
    score = 0
    actor_loss = 0
    while not done:
        action = agent.choose_action(scalers_dict['state_scalers'].transform(np.array([observation])))
        observation_,real_action, reward, done, info = env.step(action[0], index)
        score += reward
        action = scalers_dict['actions_scalers'].transform(action)
        real_action = scalers_dict['actions_scalers'].transform(real_action)
        agent.remember(scalers_dict['state_scalers'].transform(np.array([observation])), action,real_action, reward, scalers_dict['state_scalers'].transform(np.array([observation_])), done)
        # if not load_checkpoint:
        if agent.memory.mem_cntr > agent.batch_size:
            
            actor_loss += agent.learn(scalers_dict)
        else:
            actor_loss += 100
        observation = observation_
        index += 1
    actor_losses.append(actor_loss)
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    avg_loss = np.mean(actor_losses[-100:])

    if avg_loss < best_loss:
        best_loss = avg_loss
        if not load_checkpoint:
            agent.save_models()
            with open('RL_pretraining/best_score.txt', 'w') as f:
                f.write(str(best_loss))

    print('episode ', i, 'score %.1f' % actor_loss, 'avg_score %.1f' % avg_loss)

if not load_checkpoint:
            x = [i+1 for i in range(n_games)]
            plot_learning_curve(x, actor_losses, figure_file)            
            plt.show()