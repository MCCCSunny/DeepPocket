from agent.actor import Actor
from agent.critic import Critic
import torch
from torch.distributions import Categorical

class Agent():

    def __init__(self, in_channels,assets_number,gnn_parameters, trading_window_size, lr, gamma):
        self.actor = Actor(in_channels,trading_window_size,lr = lr,gnn_parameters = gnn_parameters)
        self.critic = Critic(in_channels,assets_number,trading_window_size,lr = lr)
        self.gamma = gamma

    def get_action(self,obs,prev_weigths):
        return self.actor(obs, prev_weigths)

    def learn(self,obs,next_obs,reward, weights):
        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        q_pred = self.critic(obs)
        q_next = self.critic(next_obs)
        adv = reward + self.gamma* q_next - q_pred

        critic_loss = adv.pow(2).mean()
        critic_loss.backward()
        self.critic.optimizer.step()

        dist = Categorical(weights)
        actor_loss = -(dist.log_prob(dist.sample()) * adv.detach()).mean()
        actor_loss.backward()
        self.actor.optimizer.step()
        

