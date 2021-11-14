from stock_trading_env import StockTradingEnv
import numpy as np
import random
import time
from gnn_wrapper.gnn_wrapper import GnnObservationWrapper
from aec_wrapper.autoencoder_wrapper import AecObservationWrapper

def get_action():
    values = np.array([random.uniform(0,1) for i in range(29)])
    x = (values - values.min()) / (values - values.min()).sum()
    return x

start = time.time()
env = StockTradingEnv(False,10,50,"postgresql+psycopg2://postgres:lozinka@localhost:5555/diplomski")
env = AecObservationWrapper(env,aec_layers_size = [11,[10,9]],out_features = 3,autoencoder_path='aec_wrapper/linear_autoencoder.pt')
env = GnnObservationWrapper(env,10,[3,3])


a = []
for i in range(1):
    done = False
    obs, start_weights = env.reset()
    while not done:
        new_weights = get_action()
        obs, step_reward, done, _ = env.step(new_weights)
        a.append(step_reward)


print(time.time()-start)
print(env.get_current_portfolio_value())

        





