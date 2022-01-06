import gym
from collections import deque
import numpy as np
from environment.gnn_wrapper.model import ChebNetwork
import torch
from torch_geometric.data import Data, Batch

class GnnObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, trading_window_size,input_channels,hidden_channels,output_channels,K,number_of_assets):
        super(GnnObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low,env.observation_space.high,dtype = np.float32)
        self.batch_data_deque = deque(maxlen=trading_window_size)
        self.obs_data_deque = deque(maxlen=trading_window_size)
        self.gcn = ChebNetwork(input_channels,hidden_channels,output_channels,K)
        self.edge_index = torch.tensor([np.tile(np.arange(0,number_of_assets),(number_of_assets)), np.tile(np.arange(0,number_of_assets),(number_of_assets,1)).transpose().flatten()])
        self.number_of_assets = number_of_assets
        self.trading_window_size = trading_window_size
        self.counter = 0

    # calculate mean i x j x k  -> j x k trough i 
    def mean(self,obs):
        return np.einsum('...ijk->...jk',obs)/len(obs)

    def get_inital_data_rolling(self, obs):
        data_batch = []
        for i in range(0,len(obs)):
            observation = obs[i]
            corr = abs(np.corrcoef(self.mean(obs[:i+1])))
            data_batch.append(Data(x = torch.tensor(observation, dtype = torch.float32), edge_index = self.edge_index, edge_attr = torch.tensor(corr.flatten(),dtype=torch.float32)))

        return data_batch
    
    def forward(self,batch):
        with torch.no_grad():
            x = self.gcn(batch).reshape(3,self.number_of_assets,self.trading_window_size).unsqueeze(0)
        
        return x

    def reset(self):
        obs, position = self.env.reset()
        self.obs_data_deque.extend(obs)
        data_batch = self.get_inital_data_rolling(obs)
        self.batch_data_deque.extend(data_batch)
        batch = Batch.from_data_list(data_batch)
        
        return self.forward(batch), torch.tensor(position, dtype=torch.float32)

    def observation(self, observation):
        self.obs_data_deque.append(observation)
        edge_weights = abs(np.corrcoef(self.mean(self.obs_data_deque)).flatten())
        self.batch_data_deque.append(Data(x = torch.tensor(observation, dtype = torch.float32), edge_index = self.edge_index, edge_attr = torch.tensor(edge_weights,dtype = torch.float32)))
        batch = Batch.from_data_list(self.batch_data_deque)

        return self.forward(batch)

    
    def get_model_parameters(self):
        return self.gcn.parameters()
    