import numpy as np
import torch
from torch.distributions import Geometric


class ReplayBuffer():
    def __init__(self, max_size,trading_window_size,input_dims,number_of_assets,batch_size,sample_bias = 1e-5):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.state_memory =  np.zeros((self.mem_size,input_dims[0],number_of_assets,trading_window_size), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size,number_of_assets+1), dtype=np.float32)
        self.new_state_memory =  np.zeros((self.mem_size,input_dims[0],number_of_assets,trading_window_size), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.sample_bias = sample_bias
        #probabilities = [sample_bias*pow((1-sample_bias),self.mem_size - t - self.batch_size) for t in range(self.mem_size - self.batch_size)]
        #self.distribution = Geometric(torch.tensor(probabilities))

    def store_transition(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.mem_cntr += 1

    def sample_buffer(self):
        # end = min(self.mem_cntr - self.batch_size, self.mem_size-self.batch_size)
        # index = np.random.geometric(self.sample_bias)
        # while index > end:
        #     index = np.random.geometric(self.sample_bias)

        # return self.state_memory[index:index+self.batch_size], self.action_memory[index:index+self.batch_size],  self.reward_memory[index:index+self.batch_size], self.new_state_memory[index:index+self.batch_size]
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]

        return states, actions, rewards, states_