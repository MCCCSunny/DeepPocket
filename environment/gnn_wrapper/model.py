import torch
import torch_geometric.nn as nn

class ChebNetwork(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ChebNetwork, self).__init__()
        self.conv1 = nn.ChebConv(in_channels, 64, K=2)
        self.conv2 = nn.ChebConv(64, 64, K=2)
        self.conv3 = nn.ChebConv(64, 64, K=2)
        self.conv4 = nn.ChebConv(64, out_channels, K=2)
  
    
    def forward(self, data):
        x, edge_index, edge_weight = data.x , data.edge_index, data.edge_attr
        x = torch.sigmoid(self.conv1(x, edge_index, edge_weight))
        x = torch.sigmoid(self.conv2(x, edge_index, edge_weight))
        x = torch.sigmoid(self.conv3(x, edge_index, edge_weight))
        x = torch.sigmoid(self.conv4(x, edge_index, edge_weight))
        
        return x
        