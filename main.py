import pybullet_envs
import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import sys
sys.path.append('RL_agent')
sys.path.append('Synthetic_data')
sys.path.append('Comfort_model')
sys.path.append('Environment')
sys.path.append('Energy_model')
from sac_torch import Agent
from utils import plot_learning_curve
from Environment import Environment
from Gym_Environment import MLPEnvironment
from mlp_operational_data import MLP

# test_data = pd.read_csv('Hold_out_data.csv')
# X_test = test_data.iloc[:, range(1,12)]
# y_test = test_data.iloc[:, 0]
test_data = pd.read_csv('Environment/testing_environment.csv')
X_test = test_data.iloc[:, range(11)]
y_test = test_data.iloc[:, [13,14]]
if __name__ == '__main__':
    # load all models and scalers
    models = ['Energy_model/energy_model_augmented_data.h5','Environment/model_next_state.pt','Environment/model_air_temp_suplly_return.pt','Environment/model_air_flowrate.pt' ]
    scalers = ['Energy_model/augmented_data_scaler.pkl', 'Environment/scaler_environment.pkl']
    models_dict = dict()
    scalers_dict = dict()
    for mod in models:
        model = mod.split('/')[-1].split('.')[0]
        if mod.split('/')[-1].split('.')[1] == 'pt':
            models_dict[model] = torch.load(mod).eval()
        else:
            models_dict[model] = keras.models.load_model(mod)
    for sca in scalers:
        scaler = sca.split('/')[-1].split('.')[0]
        with open(sca, 'rb') as f:
            scalers_dict[scaler] = pickle.load(f)

    X_test = scalers_dict['scaler_environment'].transform(X_test)    
    y_pred = models_dict['model_air_temp_suplly_return'](torch.tensor(X_test).float())
    # r2_score
    r2 = r2_score(y_test, y_pred.detach().numpy())
    print('R2 score: ', r2)
    env = gym.make('InvertedPendulumBulletEnv-v0')
    agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0])
    n_games = 10

    filename = 'inverted_pendulum.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
        plt.show()