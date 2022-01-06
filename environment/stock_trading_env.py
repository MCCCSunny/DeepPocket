from gym import spaces, Env
from gym.utils import seeding
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from datetime import timedelta, datetime
from math import log

class StockTradingEnv(Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, online, trading_window_size,max_buffer_size,database_url,start_date,end_date,assets_number = 28,starting_cash = 1):
        self.seed()
        self.engine = create_engine(database_url)
        self.commision_p = 0.0015 
        self.commision_s = 0.0015  
        self.assets_number = assets_number
        self.max_buffer_size = max_buffer_size
        self.trading_window_size = trading_window_size

        # spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape = (self.trading_window_size,self.assets_number + 1),dtype=np.float32)

        # episode
        self.starting_portfolio_value = starting_cash
        self.current_portfolio_value = self.starting_portfolio_value
        self.starting_date = start_date
        self.end_tick = int(pd.read_sql('select count(*) from data a where a.date >= \'{}\' and a.date <= \'{}\''.format(start_date, end_date),self.engine).iloc[0]['count'] / self.assets_number)
        self.done = False
        self.found_aprox_mi = False
        self.position = None
        self.total_reward = 0
        self.history = None
        self.mi = 0
        self.days = 0
        self.x = 0 
        
    def get_data(self,from_date):
        from_date = datetime.strptime(str(from_date),'%Y-%m-%d').date()
        starting_window_date = from_date - timedelta(days=self.trading_window_size*2)
        buffer_ending_date = from_date + timedelta(days=self.max_buffer_size+20)
        df = pd.read_sql('select * from data a where a.date >= \'{}\' and a.date <= \'{}\''.format(str(starting_window_date),str(buffer_ending_date)),self.engine)
        index = df[df['date'].ge(str(from_date))].index[0] // 28
        buffer_ending_date = df.iloc[-1]['date']
        df = df.groupby('date')
        
        return [df.get_group(x).to_numpy() for x in df.groups], index, str(buffer_ending_date)


    def reset(self):
        self.done = False
        self.position = np.concatenate(([1],np.zeros(self.assets_number)),axis = 0)
        self.total_reward = 0.0
        self.history = {}
        self.days = 0
        self.x = 0
        self.current_portfolio_value = self.starting_portfolio_value
        self.trading_buffer, self.current_tick, self.buffer_end_date = self.get_data(self.starting_date)

        return self.get_observation(reset=True), self.position


    def step(self, equity_stock_weights):
        self.days +=1
        self.current_tick += 1

        if self.days == self.end_tick:
            self.done = True
        
        if self.current_tick == len(self.trading_buffer):
            self.trading_buffer, self.current_tick, self.buffer_end_date = self.get_data(self.buffer_end_date)

        step_reward = self.calculate_reward(equity_stock_weights)
        self.position = equity_stock_weights
        self.total_reward += step_reward

        observation = self.get_observation()

        info = dict(
            portfolio_value = self.current_portfolio_value,
            reward = step_reward,
            position = equity_stock_weights
        )

        self.update_history(info)

        return observation, step_reward, self.done, info


    def get_observation(self,reset = False):
        if reset:
            return np.array(self.trading_buffer[self.current_tick-self.trading_window_size:self.current_tick]) 
        else:
            return self.trading_buffer[self.current_tick - 1]


    def update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def calculate_reward(self, stock_weights):

        y_t = np.concatenate(([1],self.trading_buffer[self.current_tick - 1][:,5].reshape(-1)), axis=0)
        
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
            if abs(self.mi -new_mi ) < 0.00001:
                self.found_aprox_mi = True

            self.mi = new_mi

        before = self.current_portfolio_value
        self.current_portfolio_value = self.current_portfolio_value *self.mi * np.dot(y_t,stock_weights)

        return log(self.current_portfolio_value/before) * 1000

    
    def get_current_portfolio_value(self):
        return self.current_portfolio_value
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]
    
    def set_dates(self,start_date,end_date):
        self.starting_date = start_date
        self.end_tick = int(pd.read_sql('select count(*) from data a where a.date >= \'{}\' and a.date <= \'{}\''.format(start_date, end_date),self.engine).iloc[0]['count'] / 28)
            