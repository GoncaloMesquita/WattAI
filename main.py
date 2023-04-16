
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
import torch
import pandas as pd
import sys
sys.path.append('RL_agent')
sys.path.append('Synthetic_data')
sys.path.append('Comfort_model')
sys.path.append('Environment')
sys.path.append('Energy_model')
from sac_torch import Agent
from utils import plot_learning_curve




if __name__ == '__main__':
    flag = 0 # 0 for training agent on simulated data, 1 for training agent on real data

    if flag == 0:
        from Environment import Environment
        from Gym_Environment import MLPEnvironment
        from mlp_operational_data import MLP
    
        data_set = pd.read_csv('dataset_building.csv')

        # load all models and scalers
        models = ['Energy_model/energy_model_augmented_data.h5','Environment/model_next_state.pt','Environment/model_air_temp_suplly_return.pt','Environment/model_air_flowrate.pt' ]
        scalers = ['Energy_model/augmented_data_scaler.pkl', 'Environment/scaler_environment.pkl', 'Scalers/state_scalers.pkl', 'Scalers/actions_scalers.pkl']
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
        
        env = MLPEnvironment(Environment, 6, 5, data_set.iloc[:-97])

        agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0], gamma=0.99,alpha = 0.003,beta = 0.003, max_size=1024*2, tau=0.001,
                    batch_size=512, reward_scale=5)
        n_games = 1000

        filename = 'plot_reward.png'

        figure_file = 'Plots/' + filename

        # best_score = env.reward_range[0]
        # extract the best score from the file best_score.txt
        with open('RL_agent/best_score.txt', 'r') as f:
            best_score = float(f.read())



        score_history = []
        load_checkpoint = False

        # if load_checkpoint:
        #     agent.load_models()
            # env.render(mode='human')
        agent.load_models()


        for i in range(n_games):
            observation, index = env.reset()
            done = False
            score = 0
            while not done:
                index += 1
                # action = agent.choose_action(observation)
                action = agent.choose_action(scalers_dict['state_scalers'].transform(np.array([observation])))
                observation_, reward, done, info = env.step(action[0], observation, models_dict, scalers_dict, data_set.iloc[index, 1])
                score += reward
                action = scalers_dict['actions_scalers'].transform(action)
                
                agent.remember(scalers_dict['state_scalers'].transform(np.array([observation])), action, reward, scalers_dict['state_scalers'].transform(np.array([observation_])), done)
                # if not load_checkpoint:
                agent.learn()
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-50:])

            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_models()
                    with open('RL_agent/best_score.txt', 'w') as f:
                        f.write(str(best_score))

            print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

        if not load_checkpoint: 
            x = [i+1 for i in range(n_games)]
            plot_learning_curve(x, score_history, figure_file)
            plt.show()


        # training on real data            
    elif flag == 1:
        from Environment_real_data import Environment_real_data
        from Gym_Environment_real_data import Real_Environment
        
        data_set_environment = pd.read_csv('Environment/data_set_environment.csv')
        
        # load all models and scalers
        scalers = ['Energy_model/augmented_data_scaler.pkl', 'Environment/scaler_environment.pkl', 'Scalers/state_scalers.pkl', 'Scalers/actions_scalers.pkl']
        scalers_dict = dict()
        for sca in scalers:
            scaler = sca.split('/')[-1].split('.')[0]
            with open(sca, 'rb') as f:
                scalers_dict[scaler] = pickle.load(f)

        env = Real_Environment(Environment_real_data, 6, 5, data_set_environment)

        agent = Agent(input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0], gamma=0.99,alpha = 0.003,beta = 0.003, max_size=1024*2, tau=0.001,
                    batch_size=512, reward_scale=5, chkpt_dir='RL_agent/sac_real_data')
        n_games = 7000

        filename = 'plot_reward.png'

        figure_file = 'Plots/' + filename

        best_score = env.reward_range[0]
        # extract the best score from the file best_score.txt
        with open('RL_agent/real_data_best_score.txt', 'r') as f:
            best_score = float(f.read())



        score_history = []
        load_checkpoint = False

        agent.load_models()


        for i in range(n_games):
            observation, index = env.reset()
            done = False
            score = 0
            while not done:
                action = agent.choose_action(scalers_dict['state_scalers'].transform(np.array([observation])))
                observation_, reward, done, info = env.step(action[0], index)
                score += reward
                action = scalers_dict['actions_scalers'].transform(action)
                
                agent.remember(scalers_dict['state_scalers'].transform(np.array([observation])), action, reward, scalers_dict['state_scalers'].transform(np.array([observation_])), done)
                # if not load_checkpoint:
                agent.learn()
                observation = observation_
                index += 1

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_models()
                    with open('RL_agent/real_data_best_score.txt', 'w') as f:
                        f.write(str(best_score))

            print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

        if not load_checkpoint:
            x = [i+1 for i in range(n_games)]
            plot_learning_curve(x, score_history, figure_file)
            plt.show()