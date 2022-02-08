import numpy as np
import torch
from torch.distributions import Geometric


class ReplayBuffer():
    def __init__(self, max_size,trading_window_size,input_dims,number_of_assets,batch_size,sample_bias = 1e-5):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.state_memory = np.zeros((self.mem_size,trading_window_size,number_of_assets,input_dims[0]), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,trading_window_size,number_of_assets,input_dims[0]), dtype=np.float32)
        self.action_memory = [None]*self.mem_size
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        #probabilities = [sample_bias*pow((1-sample_bias),self.mem_size - t - self.batch_size) for t in range(self.mem_size - self.batch_size)]
        #self.distribution = Geometric(torch.tensor(probabilities))

    def store_transition(self, state, action, reward, state_):
        # index = self.mem_cntr % self.mem_size
        self.state_memory[self.mem_cntr] = state
        self.action_memory[self.mem_cntr] = action
        self.reward_memory[self.mem_cntr] = reward
        self.new_state_memory[self.mem_cntr] = state_
        self.mem_cntr += 1

    def sample_buffer(self):
        index =0
        
        return self.state_memory[index:index+self.batch_size], self.action_memory[index:index+self.batch_size],  self.reward_memory[index:index+self.batch_size], self.new_state_memory[index:index+self.batch_size]
    
    def reset(self):
        self.mem_cntr = 0
        self.state_memory = np.zeros(self.state_memory.shape, dtype=np.float32)
        self.new_state_memory = np.zeros(self.new_state_memory.shape, dtype=np.float32)
        self.action_memory = [None]*self.mem_size
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)