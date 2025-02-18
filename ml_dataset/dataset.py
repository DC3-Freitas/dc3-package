"""
PyTorch Dataset for crystal structures
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm

class CrystalDataset(Dataset):
    def __init__(self, folder):
        self.data = []
        self.labels = []
        self.label_map = {}
        for f in tqdm(os.listdir(folder)):
            if f.endswith(".gz"):
                data = np.loadtxt(os.path.join(folder, f))
                # check data integrity
                if np.any(np.isnan(data)):
                    print(f"Skipping {f} due to NaN values")
                    continue
                label = f.split("_")[0]
                if label not in self.label_map:
                    self.label_map[label] = len(self.label_map)
                self.data.append(data)
                self.labels += [self.label_map[label] for _ in range(data.shape[0])]
        self.data = np.vstack(self.data)
        # normalize data
        means = np.mean(self.data, axis=0)
        stds = np.std(self.data, axis=0)
        self.data = (self.data - means) / stds
        self.labels = np.array(self.labels)
        print(f"Loaded dataset with {len(self.label_map)} classes and {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]