import numpy as np
from torch.distributions import Geometric
import torch
import copy

class ReplayBuffer():
    def __init__(self, max_size,batch_size,sample_bias = 1e-5):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.state_memory = []
        self.new_state_memory = [None]*self.mem_size
        self.action_memory = [None]*self.mem_size
        self.reward_memory = [None]*self.mem_size
        self.sample_bias = sample_bias

    def store_transition(self, state, action, reward, state_):
        # index = self.mem_cntr % self.mem_size
        self.state_memory[self.mem_cntr] = state
        self.action_memory[self.mem_cntr] = action
        self.reward_memory[self.mem_cntr] = reward
        self.new_state_memory[self.mem_cntr] = state_
        self.mem_cntr += 1

    def sample_buffer(self):
        ran = np.random.geometric(self.sample_bias)
        end = self.mem_cntr - self.batch_size 
        while ran > (end - 0):
            ran = np.random.geometric(self.sample_bias)
        
        index = end - ran

        return self.state_memory[index:index+self.batch_size], self.action_memory[index:index+self.batch_size],  self.reward_memory[index:index+self.batch_size], self.new_state_memory[index:index+self.batch_size]
    
    def reset(self):
        self.mem_cntr = 0
        self.state_memory = [None]*self.mem_size
        self.new_state_memory = [None]*self.mem_size
        self.action_memory = [None]*self.mem_size
        self.reward_memory = [None]*self.mem_size