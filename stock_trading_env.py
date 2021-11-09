from gym import spaces, Env
from gym.utils import seeding
import numpy as np
import torch
from sqlalchemy import create_engine
import pandas as pd
from datetime import timedelta, datetime
from autoencoder.model import LinearAutoEncoder
from math import log

class StockTradingEnv(Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, online, trading_window_size,max_buffer_size,database_url,autoencoder_path,number_of_stocks = 28, start_date = "2020-01-02", end_date = "2021-01-02",starting_cash = 2000):
        self.seed()
        self.engine = create_engine(database_url)
        self.commision_p = 0.0025 
        self.commision_s = 0.0025  
        self.number_of_stocks = number_of_stocks
        self.max_buffer_size = max_buffer_size
        self.trading_window_size = trading_window_size
        self.autoencoder = LinearAutoEncoder(in_features =  11, hidden_size = [10,9], out_features = 3)
        self.autoencoder.load_state_dict(torch.load(autoencoder_path))
        self.autoencoder.eval()

        # spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.trading_window_size,self.number_of_stocks + 1),dtype=np.float32)

        # episode
        self.starting_portfolio_value = starting_cash
        self.current_portfolio_value = self.starting_portfolio_value
        self.starting_date = datetime.strptime(start_date,'%Y-%m-%d').date()
        self.trading_buffer, self.start_tick,self.buffer_end_date = self.get_data(self.starting_date)
        self.end_tick = np.busday_count(self.starting_date, datetime.strptime(end_date,'%Y-%m-%d').date())
        self.done = False
        self.found_aprox_mi = False
        self.position = None
        self.total_reward = 0
        self.history = None
        self.mi = 0
        self.days = 0

    def get_data(self,from_date):
        from_date = datetime.strptime(str(from_date),'%Y-%m-%d').date()
        starting_window_date = from_date - timedelta(days=self.trading_window_size-5)
        buffer_ending_date = from_date + timedelta(days=self.max_buffer_size+10)
        df = pd.read_sql('select * from data a where a.date >= \'{}\' and a.date <= \'{}\''.format(str(starting_window_date),str(buffer_ending_date)),self.engine)
        df[['v1','v2','v3']] = df[['High','Low','Close','ATR','CCI','CSI','demand_index','DMI','EMA','HMA','MOM']].apply(self.encode, axis=1)
        df = df.groupby('date')
        dates_in_buffer = list(df.groups.keys())
        index = dates_in_buffer.index(str(from_date))

        return [df.get_group(x) for x in df.groups], index, dates_in_buffer[-1]


    def reset(self):
        self.done = False
        self.current_tick = self.start_tick
        self.position = np.concatenate(([1],np.zeros(self.number_of_stocks)),axis = 0)
        self.total_reward = 0.0
        self.history = {}
        self.mi = 0
        self.days = 0
        self.current_portfolio_value = self.starting_portfolio_value
        self.trading_buffer, self.current_tick, self.buffer_end_date = self.get_data(self.starting_date)

        return self.get_observation(), self.position


    def step(self, equity_stock_weights):
        self.done = False
        self.current_tick += 1
        self.days +=1
        if self.days == self.end_tick:
            self.done = True
        
        if self.current_tick == len(self.trading_buffer):
            self.trading_buffer, self.current_tick, self.buffer_end_date = self.get_data(self.buffer_end_date)

        equity_stock_weights = np.array(equity_stock_weights)
        step_reward = self.calculate_reward(equity_stock_weights)
        self.total_reward += step_reward


        observation = self.get_observation()
        info = dict(
            portfolio_value = self.current_portfolio_value,
            reward = step_reward,
            position = equity_stock_weights
        )

        self.update_history(info)

        return observation, step_reward, self.done, info


    def get_observation(self):
        return [x[['v1','v2','v3']].to_numpy() for x in self.trading_buffer[self.current_tick-self.trading_window_size:self.current_tick]]


    def update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def calculate_reward(self, stock_weights):

        y_t =  np.concatenate(([1],self.trading_buffer[self.current_tick][['Close']].to_numpy().reshape(-1)), axis=0)
        
        # Aprox mi
        if not self.found_aprox_mi:
            commision_weights = (y_t * stock_weights)/ np.dot(y_t, stock_weights)
            #Only first mi aprox
            if self.current_tick == self.trading_window_size + 1:
                self.mi = self.commision_p * sum(abs(commision_weights-stock_weights)[1:])
            else:
                pos_sum = 0
                for i in range(1,len(stock_weights)):
                    value = commision_weights[i] - self.mi * stock_weights[i]
                    if value > 0.0:
                        pos_sum = pos_sum + value

                new_mi = (1/(1- self.commision_p * stock_weights[0]))*(1 - self.commision_p * commision_weights[0] - (self.commision_s + self.commision_p - self.commision_s*self.commision_p)*pos_sum)
            
            #stop aprox if condition 
            if abs(self.mi -new_mi ) < 0.0001:
                self.found_aprox_mi = True

            self.mi = new_mi

        self.current_portfolio_value = self.current_portfolio_value *self.mi * np.dot(y_t,stock_weights)

        return log(self.mi * np.dot( y_t, stock_weights))

    
    def get_current_portfolio_value(self):
        return self.current_portfolio_value
    
    @torch.no_grad()
    def encode(self,row):
        output = self.autoencoder.encode(torch.tensor(row.values,dtype=torch.float32))

        return pd.Series([output[0].item(),output[1].item(), output[2].item()])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

