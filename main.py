from environment.stock_trading_env import StockTradingEnv
import time
from environment.gnn_wrapper.gnn_wrapper import GnnObservationWrapper
from environment.aec_wrapper.autoencoder_wrapper import AecObservationWrapper
from agent.agent import Agent
import torch
env = StockTradingEnv(False,30,100,"postgresql+psycopg2://postgres:lozinka@localhost:5555/diplomski",start_date = "2002-03-01",end_date="2009-04-16")
env = AecObservationWrapper(env,aec_layers_size = [11,[10,9]],out_features = 3,autoencoder_path='environment/aec_wrapper/linear_autoencoder.pt')
env = GnnObservationWrapper(env,30,[3,[32,32,32],3],2,number_of_assets=28)
agent = Agent(3,28,env.get_model_parameters(),30,[3,29],50,0.001,0.97,50)

a = []
start = time.time()

for i in range(15):
    done = False
    obs, weights = env.reset()
    actions = []
    adventages = []
    while not done:
        weights = agent.get_action(obs,weights)
        obs_, reward, done, _ = env.step(weights.detach().numpy())
        agent.store_transition(obs.detach(), weights, reward, obs_.detach())
        obs = obs_
        agent.learn()
        
    print(env.get_current_portfolio_value())

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

