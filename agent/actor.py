import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):

    def __init__(self,in_channels, trading_window_size, lr, gnn_parameters):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,3,kernel_size=(1,3))
        self.conv2 = nn.Conv2d(3,3, kernel_size=(1,trading_window_size-2))
        self.conv3 = nn.Conv2d(4,1,kernel_size=(1,1))
        params = list(gnn_parameters) + list(self.parameters())
        self.optimizer = optim.Adam(params,lr = lr)

    def forward(self, x, prev_weigths):
        prev_weigths = torch.tensor(prev_weigths, dtype=torch.float32)
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.cat((prev_weigths[1:].unsqueeze(-1).unsqueeze(0),x.squeeze(0)),0).unsqueeze(0)
        x = torch.tanh(self.conv3(x))
        
        x = torch.cat((prev_weigths[0].unsqueeze(0).unsqueeze(-1),x.squeeze(0).squeeze(0)))

        return torch.softmax(x,dim = 0).reshape(-1)

        

