import pandas as pd
import os
from torch.utils.data import Dataset
import numpy as np

class StockDataset(Dataset):
    def __init__(self,folder_path,skip_names,last_n_entries = 2500):
        data = []
        for file in os.listdir(folder_path):
            if file not in skip_names:            
                pom = pd.read_pickle(folder_path+'/'+file)
                pom.drop(columns=['date','Open','Volume'], inplace=True)
                pom = pom[-last_n_entries:]
                data.extend(pom.to_numpy())

        self.X = np.array(data, dtype=np.float32)
        self.y = self.X[:,:3]
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])
