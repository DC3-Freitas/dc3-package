"""
PyTorch Dataset for crystal structures
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class CrystalDataset(Dataset):
    def __init__(self, folder):
        self.data = []
        self.labels = []
        self.label_map = {}
        for f in os.listdir(folder):
            if f.endswith(".gz"):
                data = np.loadtxt(os.path.join(folder, f))
                label = f.split("_")[0]
                if label not in self.label_map:
                    self.label_map[label] = len(self.label_map)
                self.data += [data]
                self.labels += [self.label_map[label] for _ in range(data.shape[0])]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]