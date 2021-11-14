import gym
from collections import deque
import numpy as np
from gnn_wrapper.model import ChebNetwork
import torch
from torch_geometric.data import Data, Batch

class GnnObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, trading_window_size,gnn_layers_size):
        super(GnnObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low,env.observation_space.high,dtype = np.float32)
        self.data_deque = deque(maxlen = trading_window_size)
        self.gcn = ChebNetwork(gnn_layers_size[0],gnn_layers_size[1])
        self.edge_index = torch.tensor([np.tile(np.arange(0,28),(28)), np.tile(np.arange(0,28),(28,1)).transpose().flatten()])

    def calculate_corr(self, obs):
        corr_list = []
        for x in obs:
            corr = np.corrcoef(x)
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

        return self.gcn(batch), position

    def observation(self, observation):
        corr = np.corrcoef(observation).flatten()
        observation = torch.tensor(observation, dtype = torch.float32)
        weights = torch.tensor(corr, dtype = torch.float32)
        self.data_deque.append(Data(x = observation, edge_index = self.edge_index, edge_attr = weights))
        batch = Batch.from_data_list(list(self.data_deque))
        
        return self.gcn(batch)
    
    def get_model_parameters(self):
        return self.gcn.get_model_parameters()