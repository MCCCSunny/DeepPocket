from agent.actor import Actor
from agent.critic import Critic
import torch
from agent.replay_memory import ReplayBuffer
from agent.model import ChebNetwork
from torch.distributions import Normal
import numpy as np
import torch.optim as optim
import time

class Agent():

    def __init__(self,gnn_in_channels,gnn_hidden_channels,gnn_output_channels,cheb_k,assets_number, trading_window_size,actor_lr,critic_lr,actor_weight_decay,critic_weight_decay,gamma,batch_size,mem_size,input_dims,sample_bias,nb):
        self.actor = Actor(gnn_output_channels,trading_window_size)
        self.critic = Critic(gnn_output_channels,assets_number,trading_window_size,lr = critic_lr,weight_decay = critic_weight_decay)
        gnn_hidden_channels = [int(x) for x in gnn_hidden_channels.split(',')]
        self.gcn = ChebNetwork(gnn_in_channels,gnn_hidden_channels,gnn_output_channels,cheb_k)
        self.gamma = gamma
        self.batch_size = batch_size
        self.trading_window_size = trading_window_size
        self.assets_number = assets_number
        self.memory = ReplayBuffer(mem_size,batch_size,sample_bias)
        self.nb = nb
        params = list(self.actor.parameters()) + list(self.gcn.parameters()) 
        self.actor_gnn_optimizer = optim.Adam(params,lr = actor_lr,weight_decay = actor_weight_decay)


    def get_action(self,obs,prev_weigths):
        obs = self.gnn_forward(obs)
        return self.actor_forward(obs, prev_weigths)
    
    def actor_forward(self,obs,prev_weigths):
        return self.actor(obs.unsqueeze(0), prev_weigths)

    def gnn_forward(self,batch):
        return self.gcn(batch).reshape(3,self.assets_number,self.trading_window_size)
    
    def store_transition(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def sample_memory(self):
        state, actions, reward, new_state = self.memory.sample_buffer()
        states = state
        actions = torch.stack(actions)
        rewards = torch.tensor(reward)
        states_ = new_state

        return states, actions, rewards, states_
    
    def learn(self):
        if self.memory.mem_cntr %100 != 0:
            return
        start = time.time()
        self.critic.optimizer.zero_grad()
        self.actor_gnn_optimizer.zero_grad()
        actor_loss = 0
        critic_loss = 0
        for _ in range(self.nb):
            states, prev_weights, rewards, states_, = self.sample_memory()
            actions = []
            gnn_states = []
            gnn_states_ = []

            for batch,prev_weight,batch_ in zip(states,prev_weights,states_):
                current = self.gnn_forward(batch)
                gnn_states.append(current.detach())
                actions.append(self.actor_forward(current,prev_weight))
                gnn_states_.append(self.gnn_forward(batch_).detach())
            
                
            gnn_states = torch.stack(gnn_states)
            gnn_states_ = torch.stack(gnn_states_)
            q_pred = self.critic(gnn_states)
            q_next = self.critic(gnn_states_)
            adv =  rewards + self.gamma* q_next - q_pred
            critic_loss = torch.mean(adv * q_pred)
            
            actions = torch.stack(actions)
            # x = actions.detach()

            # mean,std  = torch.mean(x, dim=0), torch.std(x, dim = 0)

            # mean = torch.clip(mean, min = 1e-6, max = 60)
            # std = torch.clip(std,min = 1e-6,max = 30)
            # dist = Normal(mean,std)
            actor_loss = torch.mean(-1*torch.mul(torch.mean(torch.log(actions)),adv.detach()))
            #actor_loss += torch.mean(dist.log_prob(actions).sum() * adv.clone().detach())
            #actor_loss = (actions.log().sum()*adv.detach()).sum()
            actor_loss.backward()
            critic_loss.backward()

            self.critic.optimizer.step()
            self.actor_gnn_optimizer.step()
        print(time.time() - start)

    def reset_memory(self):
        self.memory.reset()
    
        
