import time

from numpy import dtype
from agent.agent import Agent
from environment.utils import make_env
import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def train(env,agent,num_episodes,args):
    a = []
    start = time.time()
    fig, ax = plt.subplots()
    with torch.autograd.set_detect_anomaly(True):
        for i in tqdm(range(num_episodes)):
            done = False
            env.set_dates(args.train_start_date,args.train_end_date)
            obs, weights = env.reset()
            while not done:
                next_weights = agent.get_action(obs,weights).detach()
                obs_, reward, done, _ = env.step(next_weights.numpy())
                # pv_vector = torch.sum(weights*torch.from_numpy(np.array(env.future_prices,dtype=np.float16)))
                agent.store_transition(obs, weights, reward, obs_ )
                agent.learn()
                obs = obs_
                weights = next_weights
                a.append(env.get_current_portfolio_value())
            print(min(a),a[-1],max(a))
            if i % 100 == 0:
                ax.plot(range(len(a)),a)
                ax.figure.savefig('./plots/train_'+str(i))
                test(env,agent,args.test_start_date,args.test_end_date,i)
            
            a = []

    print('Training time:',time.time() - start)

@torch.no_grad()
def test(env,agent,start_date,end_date,epoch):
    done = False
    env.set_dates(start_date,end_date)
    obs, weights = env.reset()
    a = []
    all_weights = []
    
    while not done:
        with torch.no_grad():
            #all_weights.append(weights.numpy())
            weights = agent.get_action(obs,weights)
            obs_, reward, done, _ = env.step(weights.detach().numpy())
            obs = obs_
            a.append(env.get_current_portfolio_value())
    test_ax.plot(range(len(a)),a)
    test_ax.figure.savefig('./plots/test_'+str(epoch))

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--assets_number', type = int, default = 28, help='number of assets')
    parser.add_argument('--trading_window_size',type = int, default = 30, help= 'number of last n trades taking in consideration')
    parser.add_argument('--gamma', type = float, default = 0.99, help='discount factor')
    parser.add_argument('--device', type=str, default='cpu', help='gpu/cpu')
    parser.add_argument('--num_episodes', type=int, default=1000, help='number of training episodes')    
    parser.add_argument('--batch_size', type=int, default=109, help='batch size') 
    parser.add_argument('--actor_lr', type=float, default=1e-5, help='actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=2e-3, help='critic learning rate')
    parser.add_argument('--actor_weight_decay', type=float, default=1e-8, help='L2 regularization on actor model weights')
    parser.add_argument('--critic_weight_decay', type=float, default=1e-8, help='L2 regularization on critic model weights')
    parser.add_argument('--train_start_date', type=str,default = '2002-04-01', help='training start date (format: %YYYY-mm-dd)')
    parser.add_argument('--train_end_date', type=str,default = '2018-10-19', help='training end date (format: %YYYY-mm-dd)')
    parser.add_argument('--test_start_date', type=str,default = '2019-06-09', help='testing start date (format: %YYYY-mm-dd)')
    parser.add_argument('--test_end_date', type=str,default = '2019-10-16', help='testing end date (format: %YYYY-mm-dd)')
    parser.add_argument('--starting_cash',type = float, default = 1, help='starting cash value')
    parser.add_argument('--online',type = bool,default = False, help='offline/online usage')
    parser.add_argument('--buffer_size', type = int, default = 400,help='size of buffer containg data')
    parser.add_argument('--database_url',type = str, default = 'postgresql+psycopg2://postgres:lozinka@localhost:5555/diplomski', help='database url to save/load data')
    parser.add_argument('--cheb_k', type = int, default = 2, help = 'size of cheb filter')
    parser.add_argument('--gnn_input_channels',type = int, default = 3, help = 'gnn input channels')
    parser.add_argument('--gnn_hidden_channels',type = str,default='8,8,8', help = 'hidden channel sizes (format: 16,16,16)')
    parser.add_argument('--gnn_output_channels',type = int, default = 3, help = 'gnn output channels')
    parser.add_argument('--mem_size',type = int, default = 5000, help = 'memory size')
    parser.add_argument('--sample_bias', type=float, default = 5e-5,help = 'sample bias')
    parser.add_argument('--number_of_batches', type=int, default = 3, help = 'number of minibatches in agent learning')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    env = make_env(args)
    test_fig, test_ax = plt.subplots()
    agent = Agent(args.gnn_input_channels,args.gnn_hidden_channels,args.gnn_output_channels,args.cheb_k,args.assets_number,args.trading_window_size,args.actor_lr,args.critic_lr,args.actor_weight_decay,args.critic_weight_decay,args.gamma,args.batch_size,args.mem_size,[args.gnn_output_channels,args.assets_number+1],args.sample_bias, args.number_of_batches)
    train(env,agent,args.num_episodes,args)
    test(env,agent,args.test_start_date,args.test_end_date,'last')