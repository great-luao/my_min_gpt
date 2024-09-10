import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

# Make tok_train_df into Dataset
class GPTDataset(Dataset):
    def __init__(self, dataframe, block_size):
        self.data = dataframe
        self.block_size = block_size

    def __getitem__(self, idx):
        review = self.data[idx]
        ix = torch.randint(len(review) - self.block_size, (1,))
        x = torch.from_numpy(review[ix:ix+self.block_size].astype(np.int64))
        y = torch.from_numpy(review[ix+1:ix+self.block_size+1].astype(np.int64))
        return x,y 

    def __len__(self):
        # the number of reviews in the dataset
        return len(self.data)