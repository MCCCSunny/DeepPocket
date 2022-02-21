import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):

    def __init__(self,in_channels,number_of_assets,trading_window_size,actor_lr,actor_weight_decay):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,3,kernel_size=(1,3))
        #self.batch_norm1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3,3, kernel_size=(1,trading_window_size-2))
        #self.batch_norm2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(4,1,kernel_size=(1,1))
        self.number_of_assets = number_of_assets
        self.trading_window_size = trading_window_size
        self.alpha = nn.Parameter(torch.tensor(0.0),requires_grad = True)
        self.optimizer = optim.Adam(self.parameters(),lr = actor_lr,weight_decay = actor_weight_decay)

    def forward(self, x, prev_weigths, learn = False):
        prev_weigths = prev_weigths.clone().detach()
        if learn == False:
            x = x.unsqueeze(0)
        x = torch.tanh(self.conv1(x))
        #x = self.batch_norm1(x)
        x = torch.tanh(self.conv2(x))
        #x = self.batch_norm2(x)
        
        if learn:
            c = prev_weigths[:,1:].unsqueeze(-1).unsqueeze(0).permute(1,0,2,3)
            x = torch.cat([c,x],dim = 1)
        else:
            x = torch.cat((prev_weigths[1:].unsqueeze(-1).unsqueeze(0),x.squeeze(0)),0).unsqueeze(0)
        
        x = torch.tanh(self.conv3(x))
        if learn:
            cash = torch.ones((32,1,1,1)) * self.alpha
            x = torch.cat([cash, x], -2)
            
        else:
            x = torch.cat((torch.tensor(self.alpha).unsqueeze(0).unsqueeze(-1),x.squeeze(0).squeeze(0)))

        return torch.softmax(x,dim = 0).reshape(-1)

        

