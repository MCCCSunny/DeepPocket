from stock_trading_env import StockTradingEnv
import numpy as np
import random


def get_action():
    values = np.array([random.uniform(0,1) for i in range(29)])
    x = (values - values.min()) / (values - values.min()).sum()
    return x
    
env = StockTradingEnv(False,10,100,"postgresql+psycopg2://postgres:lozinka@localhost:5555/diplomski",autoencoder_path='/home/niko/diplomski_rad/autoencoder/linear_autoencoder.pt')

for i in range(1):
    done = False
    obs, start_weights = env.reset()
    while not done:
        new_weights = get_action()
        obs, step_reward, done, _ = env.step(new_weights)
        print('Reward', step_reward)
    
print(env.get_current_portfolio_value())

        





