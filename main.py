from environment.stock_trading_env import StockTradingEnv
import time
from environment.gnn_wrapper.gnn_wrapper import GnnObservationWrapper
from environment.aec_wrapper.autoencoder_wrapper import AecObservationWrapper
from agent.agent import Agent

env = StockTradingEnv(False,50,500,"postgresql+psycopg2://postgres:lozinka@localhost:5555/diplomski")
env = AecObservationWrapper(env,aec_layers_size = [11,[10,9]],out_features = 3,autoencoder_path='environment/aec_wrapper/linear_autoencoder.pt')
env = GnnObservationWrapper(env,50,[3,[64,64,64],3],2,number_of_assets=28)
agent = Agent(3,28,env.get_model_parameters(),50,0.001,0.95)

a = []
start = time.time()
counter = 0

for i in range(1):
    done = False
    obs, weights = env.reset()
    while not done:
        weights = agent.get_action(obs,weights)
        obs_, reward, done, _ = env.step(weights)
        agent.learn(obs,obs_,reward,weights)
        obs = obs_
        a.append(reward)

print(time.time()-start)
print(env.end_tick)
print(env.get_current_portfolio_value())

        





