import gym
import numpy as np
import torch
from environment.aec_wrapper.autoencoder.model import LinearAutoEncoder
from collections import deque

class AecObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, aec_layers_size, out_features, autoencoder_path, number_of_assets, trading_window_size):
        super(AecObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low,env.observation_space.high,dtype = np.float32)
        self.autoencoder = LinearAutoEncoder(in_features =  aec_layers_size[0], hidden_size = aec_layers_size[1], out_features = out_features)
        self.filter_indices = [2,3,4,6,7,8,9,10,11,12,13,]
        self.obs_data_deque = deque(maxlen=trading_window_size)
        self.number_of_assets = number_of_assets
        self.trading_window_size = trading_window_size
        self.autoencoder.load_state_dict(torch.load(autoencoder_path))
        self.autoencoder.eval()
      
    
    def reset(self):
        obs, weights = self.env.reset()
        filtered_obs = obs[:,:,self.filter_indices].astype(np.float32)
        
        self.obs_data_deque = deque(maxlen=self.trading_window_size)
        self.obs_data_deque.extend(filtered_obs)
        with torch.no_grad():
            x = self.autoencoder.encode(torch.tensor(np.array(self.obs_data_deque),dtype=torch.float32))

        return x.reshape(3,self.number_of_assets,self.trading_window_size), torch.from_numpy(weights)
    
    def observation(self, observation):
        filtered_obs = observation[:,self.filter_indices].astype(np.float32)
        self.obs_data_deque.append(filtered_obs)

        with torch.no_grad():
            x = self.autoencoder.encode(torch.tensor(np.array(self.obs_data_deque),dtype=torch.float32))

        return x.reshape(3,self.number_of_assets,self.trading_window_size)