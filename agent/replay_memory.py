import numpy as np
import torch


class ReplayBuffer():
    def __init__(self, max_size,trading_window_size,input_dims,number_of_assets):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size,input_dims[0],number_of_assets,trading_window_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,input_dims[0],number_of_assets,trading_window_size), dtype=np.float32)
        self.action_memory = [None]*self.mem_size
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.mem_cntr += 1

    def sample_buffer(self):
        return self.state_memory, self.action_memory,  self.reward_memory, self.new_state_memory
    
    def reset(self):
        self.mem_cntr = 0
        self.state_memory = np.zeros(self.state_memory.shape, dtype=np.float32)
        self.new_state_memory = np.zeros(self.new_state_memory.shape, dtype=np.float32)
        self.action_memory = [None]*self.mem_size
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)