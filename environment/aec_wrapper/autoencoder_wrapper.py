import gym
import numpy as np
import torch
from environment.aec_wrapper.autoencoder.model import LinearAutoEncoder

class AecObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, aec_layers_size, out_features, autoencoder_path):
        super(AecObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low,env.observation_space.high,dtype = np.float32)
        self.autoencoder = LinearAutoEncoder(in_features =  aec_layers_size[0], hidden_size = aec_layers_size[1], out_features = out_features)
        self.filter_indices = [3,4,5,7,8,9,10,11,12,13,14,]
        self.autoencoder.load_state_dict(torch.load(autoencoder_path))
        self.autoencoder.eval()
    
    def reset(self):
        obs, weights = self.env.reset()
        filtered_obs = obs[:,:,self.filter_indices].astype(np.float32)
        with torch.no_grad():
            x = self.autoencoder.encode(torch.tensor(filtered_obs,dtype=torch.float32))

        return x.numpy(), weights
    
    def observation(self, observation):
        filtered_obs = observation[:,self.filter_indices].astype(np.float32)
        with torch.no_grad():
            x = self.autoencoder.encode(torch.tensor(filtered_obs,dtype=torch.float32))

        return x.numpy()
