import torch
import torch_geometric.nn as nn

class ChebNetwork(torch.nn.Module):
    def __init__(self,in_channels,hidden_channels,out_channels,K):
        super(ChebNetwork, self).__init__()
        self.conv1 = nn.ChebConv(in_channels, hidden_channels[0], K=K)
        self.conv2 = nn.ChebConv(hidden_channels[0], hidden_channels[1], K=K)
        self.conv3 = nn.ChebConv(hidden_channels[1], hidden_channels[2], K=K)
        self.conv4 = nn.ChebConv(hidden_channels[2], out_channels, K=K)

    
    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = torch.sigmoid(self.conv1(x, edge_index, edge_weight))
        x = torch.sigmoid(self.conv2(x, edge_index, edge_weight))
        x = torch.sigmoid(self.conv3(x, edge_index, edge_weight))
        x = torch.sigmoid(self.conv4(x, edge_index, edge_weight))

        return x
        