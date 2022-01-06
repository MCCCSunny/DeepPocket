from environment.stock_trading_env import StockTradingEnv
from environment.aec_wrapper.autoencoder_wrapper import AecObservationWrapper

def make_env(args):
    env = StockTradingEnv(args.online,args.trading_window_size,args.buffer_size,args.database_url,start_date = args.train_start_date,end_date=args.train_end_date,assets_number=args.assets_number,starting_cash=args.starting_cash)
    env = AecObservationWrapper(env,aec_layers_size = [11,[10,9]],out_features = 3,autoencoder_path='environment/aec_wrapper/linear_autoencoder.pt',number_of_assets= args.assets_number,trading_window_size=args.trading_window_size)

    return env