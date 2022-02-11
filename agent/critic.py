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
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1))
        self.conv2 = nn.Conv2d(in_channels,trading_window_size,kernel_size=(1,1))
        self.conv3 = nn.Conv2d(trading_window_size,1,kernel_size=(1,trading_window_size))
        self.dense = nn.Linear(num_assets,1)
        self.optimizer = optim.Adam(self.parameters(),lr = lr,weight_decay= weight_decay)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dense(x.squeeze(-1))

        return x.reshape(-1)
