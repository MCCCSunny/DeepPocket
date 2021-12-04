import time
from agent.agent import Agent
from environment.utils import make_env
import torch
import argparse


def train(env,agent,num_episodes):
    a = []
    start = time.time()
    for i in range(num_episodes):
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

def test(env,agent,start_date,end_date):
    done = False
    env.set_dates(start_date,end_date)
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

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--assets_number', type = int, default = 28, help='number of assets')
    parser.add_argument('--trading_window_size',type = int, default = 50, help= 'number of last n trades taking in consideration')
    parser.add_argument('--gamma', type = float, default = 0.97, help='discount factor')
    parser.add_argument('--device', type=str, default='cpu', help='gpu/cpu')
    parser.add_argument('--num_episodes', type=int, default=100, help='number of training episodes')    
    parser.add_argument('--batch_size', type=int, default=50, help='batch size') 
    parser.add_argument('--actor_lr', type=float, default=1e-4, help='actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=1e-4, help='critic learning rate')
    parser.add_argument('--actor_weight_decay', type=float, default=2e-8, help='L2 regularization on actor model weights')
    parser.add_argument('--critic_weight_decay', type=float, default=0, help='L2 regularization on critic model weights')
    parser.add_argument('--train_start_date', type=str,default = '2002-04-01', help='training start date (format: %YYYY-mm-dd)')
    parser.add_argument('--train_end_date', type=str,default = '2009-04-16', help='training end date (format: %YYYY-mm-dd)')
    parser.add_argument('--test_start_date', type=str,default = '2010-03-15', help='testing start date (format: %YYYY-mm-dd)')
    parser.add_argument('--test_end_date', type=str,default = '2010-07-21', help='testing end date (format: %YYYY-mm-dd)')
    parser.add_argument('--starting_cash',type = float, default = 1, help='starting cash value')
    parser.add_argument('--online',type = bool,default = False, help='offline/online usage')
    parser.add_argument('--buffer_size', type = int, default = 400,help='size of buffer containg data')
    parser.add_argument('--database_url',type = str, default = 'postgresql+psycopg2://postgres:lozinka@localhost:5555/diplomski', help='database url to save/load data')
    parser.add_argument('--cheb_k', type = int, default = 3, help = 'size of cheb filter')
    parser.add_argument('--gnn_input_channels',type = int, default = 3, help = 'gnn input channels')
    parser.add_argument('--gnn_hidden_channels',type = str,default='16,16,16', help = 'hidden channel sizes (format: 16,16,16)')
    parser.add_argument('--gnn_output_channels',type = int, default = 3, help = 'gnn output channels')
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = get_args()
    env = make_env(args)
    agent = Agent(args.gnn_output_channels,args.assets_number,env.get_model_parameters(),args.trading_window_size,args.actor_lr,args.critic_lr,args.gamma,args.actor_weight_decay,args.critic_weight_decay,args.batch_size,args.batch_size,[args.gnn_output_channels,args.assets_number+1])
    train(env,agent,args.num_episodes)
    test(env,agent,args.test_start_date,args.test_end_date)
