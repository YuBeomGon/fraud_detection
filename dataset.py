import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CDataset(Dataset):
    def __init__(self, df, eval_mode=False):
        self.df = df
        self.eval_mode = eval_mode
        # Pandas is 18 times slower than Numpy (15.8ms vs 0.874 ms). Pandas is 20 times slower than Numpy
        # therefore Use numpy for faster data loading        
        if self.eval_mode:
            self.labels = self.df['Class'].values
            self.df = self.df.drop(columns=['Class']).values
        else:
            self.df = self.df.values
        
    def __getitem__(self, index):
        if self.eval_mode:
            self.x = self.df[index]
            self.y = self.labels[index]
            return torch.Tensor(self.x), self.y
        else:
            self.x = self.df[index]
            return torch.Tensor(self.x)
        
    def __len__(self):
        return len(self.df)