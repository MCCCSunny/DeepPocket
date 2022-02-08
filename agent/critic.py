import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.data import Data, Batch
import torch
from collections import deque

from agent.model import ChebNetwork

class Critic(nn.Module):

    def __init__(self,in_channels,gnn_in_channels,gnn_hidden_channels,gnn_out_channels,cheb_k,num_assets,trading_window_size,lr, weight_decay,batch_size=32):
        super(Critic, self).__init__()
        gnn_hidden_channels = [int(x) for x in gnn_hidden_channels.split(',')]
        self.gnn = ChebNetwork(gnn_in_channels,gnn_hidden_channels,gnn_out_channels,cheb_k)
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1))
        self.conv2 = nn.Conv2d(in_channels,trading_window_size,kernel_size=(1,1))
        self.conv3 = nn.Conv2d(trading_window_size,1,kernel_size=(1,trading_window_size))
        self.dense = nn.Linear(num_assets,1)
        self.trading_window_size = trading_window_size
        self.assets_number = num_assets
        self.batch_size = batch_size
        self.edge_index = torch.tensor([np.tile(np.arange(0,num_assets),(num_assets)), np.tile(np.arange(0,num_assets),(num_assets,1)).transpose().flatten()])
        self.optimizer = optim.Adam(self.parameters(),lr = lr,weight_decay= weight_decay)

    def forward(self,x):
        #x = self.get_batch_rolling(x)
        x = F.relu(self.conv1(x.reshape(self.batch_size,3,self.assets_number,self.trading_window_size)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dense(x.squeeze(-1))

        return x.reshape(-1)

    def mean(self,obs):
        return np.einsum('...ijk->...jk',obs)/len(obs)

    def get_batch_rolling(self, obs):
        data_batch = deque(maxlen= self.trading_window_size) 
        result = []
        for i in range(0,len(obs[0])):
            observation = obs[0][i]
            corr = abs(np.corrcoef(self.mean(obs[0][:i+1])))
            data_batch.append(Data(x = observation.clone(), edge_index = self.edge_index, edge_attr = torch.tensor(corr.flatten(),dtype=torch.float32)))
        result.append(self.gnn(Batch.from_data_list(data_batch)).reshape(3,self.assets_number,self.trading_window_size))
        
        for i in range(1,len(obs)):
            observation = obs[i][-1]
            corr = abs(np.corrcoef(self.mean(obs[i])))
            data_batch.append(Data(x = observation.clone(), edge_index = self.edge_index, edge_attr = torch.tensor(corr.flatten(),dtype=torch.float32)))
            result.append(self.gnn(Batch.from_data_list(data_batch)).reshape(3,self.assets_number,self.trading_window_size))
        
        return torch.stack(result)