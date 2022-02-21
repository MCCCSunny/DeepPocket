from agent.actor import Actor
from agent.critic import Critic
import torch
from agent.replay_memory import ReplayBuffer
import numpy as np
from torch.distributions import Normal

class Agent():

    def __init__(self,gnn_in_channels,gnn_hidden_channels,gnn_output_channels,cheb_k,assets_number, trading_window_size,actor_lr,critic_lr,actor_weight_decay,critic_weight_decay,gamma,batch_size,mem_size,input_dims,sample_bias,nb):
        self.actor = Actor(gnn_output_channels,assets_number,trading_window_size,actor_lr = actor_lr, actor_weight_decay = actor_weight_decay)
        self.critic = Critic(gnn_output_channels,gnn_in_channels,gnn_hidden_channels,gnn_output_channels,cheb_k,assets_number,trading_window_size,lr = critic_lr,weight_decay = critic_weight_decay,batch_size = batch_size)
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size,trading_window_size, input_dims,assets_number,batch_size,sample_bias)
        self.nb = nb


    def get_action(self,obs,prev_weigths, learn = False):
        return self.actor(obs, prev_weigths,learn = learn)
    
    def store_transition(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def sample_memory(self):
        state, actions, reward, new_state = self.memory.sample_buffer()
        states = torch.stack(state)
        rewards = torch.tensor(reward)
        states_ = torch.stack(new_state)
        actions = torch.stack(actions)

        return states, actions, rewards, states_
    
    def learn(self):
        if self.memory.mem_cntr%51 != 0 :
            return

        
        for _ in range(self.nb):
            self.critic.optimizer.zero_grad()
            self.actor.optimizer.zero_grad()
            states, actions, rewards, states_, = self.sample_memory()
            q_pred = self.critic(states)
            q_next = self.critic(states_)
            adv =  rewards + self.gamma* q_next.detach() - q_pred.detach()

            critic_loss = torch.mean(torch.mul(adv,q_pred))
            # x = actions.clone().detach()

            # mean,std  = torch.mean(x, dim=0), torch.std(x, dim = 0)

            # mean = torch.clip(mean, min = 1e-6, max = 60)
            # std = torch.clip(std,min = 1e-6,max = 30)
            # dist = Normal(mean,std)
            
            actions = self.get_action(states,actions,learn = True)
            actor_loss = -1*torch.mean(torch.log(torch.mean(actions))*adv.detach())
            #actor_loss = -1*torch.mean(torch.sum(dist.log_prob(actions)) * adv.clone().detach())
            #actor_loss += (actions.log().mean()*adv.detach()).mean()
            
            actor_loss.backward()
            critic_loss.backward()
            self.critic.optimizer.step()
            self.actor.optimizer.step()

    
    def reset(self):
        self.memory.reset()
        


