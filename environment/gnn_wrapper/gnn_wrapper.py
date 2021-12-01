import gym
from collections import deque
import numpy as np
from environment.gnn_wrapper.model import ChebNetwork
import torch
from torch_geometric.data import Data, Batch
from scipy import signal

class GnnObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, trading_window_size,gnn_layers_size,K,number_of_assets):
        super(GnnObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low,env.observation_space.high,dtype = np.float32)
        self.data_deque = deque(maxlen = trading_window_size)
        self.gcn = ChebNetwork(gnn_layers_size[0],gnn_layers_size[1],gnn_layers_size[2],K)
        self.edge_index = torch.tensor([np.tile(np.arange(0,number_of_assets),(number_of_assets)), np.tile(np.arange(0,number_of_assets),(number_of_assets,1)).transpose().flatten()])
        self.number_of_assets = number_of_assets
        self.trading_window_size = trading_window_size
        self.counter = 0

    def calculate_corr(self, obs):
        corr_list = []
        for i in range(0,len(obs)):
            corr = 1-np.corrcoef(obs[i])
            corr_list.append(corr.flatten())

        return corr_list
        
    def reset(self):
        obs, position = self.env.reset()
        weights = self.calculate_corr(obs)
        x = torch.tensor(obs, dtype = torch.float32)
        weights = torch.tensor(weights, dtype = torch.float32)
        data_list = [Data(x = x[i], edge_index = self.edge_index, edge_attr = weights[i]) for i in range(len(x))]
        self.data_deque.extend(data_list)
        batch = Batch.from_data_list(list(self.data_deque))

        return self.gcn(batch).reshape(self.number_of_assets,self.trading_window_size,3).permute(2, 0, 1).unsqueeze(0), position

    def observation(self, observation):
        corr = np.corrcoef(observation).flatten()
        observation = torch.tensor(observation, dtype = torch.float32)
        weights = torch.tensor(1-corr, dtype = torch.float32)
        self.data_deque.append(Data(x = observation, edge_index = self.edge_index, edge_attr = weights))
        batch = Batch.from_data_list(list(self.data_deque))

        return self.gcn(batch).reshape(3,self.number_of_assets,self.trading_window_size).unsqueeze(0)

    
    def get_model_parameters(self):
        return self.gcn.parameters()