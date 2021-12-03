from environment.stock_trading_env import StockTradingEnv
import time
from environment.gnn_wrapper.gnn_wrapper import GnnObservationWrapper
from environment.aec_wrapper.autoencoder_wrapper import AecObservationWrapper
from agent.agent import Agent
import torch

env = StockTradingEnv(False,20,100,"postgresql+psycopg2://postgres:lozinka@localhost:5555/diplomski",start_date = "2002-03-01",end_date="2009-04-16")
env = AecObservationWrapper(env,aec_layers_size = [11,[10,9]],out_features = 3,autoencoder_path='environment/aec_wrapper/linear_autoencoder.pt')
env = GnnObservationWrapper(env,20,[3,[8,8,8],3],3,number_of_assets=28)
agent = Agent(3,28,env.get_model_parameters(),20,[3,29],25,1e-3,0.97,25)

a = []
start = time.time()
for i in range(100):
    done = False
    obs, weights = env.reset()
    while not done:
        weights = agent.get_action(obs,weights)
        obs_, reward, done, _ = env.step(weights.detach().numpy())
        agent.store_transition(obs.detach(), weights, reward, obs_.detach())
        obs = obs_
        agent.learn()
        a.append(env.get_current_portfolio_value())
        
        
    print(min(a),env.get_current_portfolio_value(),max(a))
    a = []

print('Training time:',time.time() - start)
done = False
env.set_dates('2010-03-15','2010-07-21')
obs, weights = env.reset()
start = time.time()
while not done:
    with torch.no_grad():
        weights = agent.get_action(obs,weights)
        obs_, reward, done, _ = env.step(weights.detach().numpy())
        obs = obs_
    print(env.get_current_portfolio_value())
    #a.append(reward)

print('Test time:',time.time() - start)

