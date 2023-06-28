from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class ARDataset(Dataset):
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        inputs = self.data.iloc[idx, 0:6]
        out = self.data.iloc[idx, 6:]

        return (torch.tensor(inputs).float(), torch.tensor(out).float())


