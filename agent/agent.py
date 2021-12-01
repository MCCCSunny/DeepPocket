from agent.actor import Actor
from agent.critic import Critic
import torch
from agent.replay_memory import ReplayBuffer
import numpy as np
from torch.distributions import Normal

class Agent():

    def __init__(self, in_channels,assets_number,gnn_parameters, trading_window_size,input_dims,mem_size, lr, gamma,batch_size):
        self.actor = Actor(in_channels,trading_window_size,lr = lr,gnn_parameters = gnn_parameters)
        self.critic = Critic(in_channels,assets_number,trading_window_size,lr = lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.criterion = torch.nn.MSELoss()
        self.memory = ReplayBuffer(mem_size,trading_window_size, input_dims,assets_number)


    def get_action(self,obs,prev_weigths):
        return self.actor(obs, prev_weigths)
    
    def store_transition(self, state, action, reward, state_):
        self.memory.store_transition(state, action, reward, state_)

    def sample_memory(self):
        state, actions, reward, new_state = self.memory.sample_buffer()
        states = torch.tensor(state)
        rewards = torch.tensor(reward)
        states_ = torch.tensor(new_state)

        return states, actions, rewards, states_
    
    def learn(self):
        if self.memory.mem_cntr <= self.batch_size:
            return
        states, actions, rewards, states_, = self.sample_memory()
        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        q_pred = self.critic(states)
        q_next = self.critic(states_)
        target_value = rewards + self.gamma* q_next
        adv =  target_value - q_pred
        critic_loss = torch.dot(adv,q_pred)
        critic_loss.backward()

        self.critic.optimizer.step()
        x = [action.detach() for action in actions]
        x = torch.stack(x)

        mean = torch.mean(x, dim=0)
        if self.batch_size == 1:
            std = mean
        else:
            std = torch.std(x, dim = 0)

        dist = Normal(mean,std)
        actor_loss = 0 
        for x,y in zip(actions,adv):
            actor_loss += -1*dist.log_prob(x) * y.detach()

        actor_loss = torch.sum(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        self.memory.reset()
        


