"""
PyTorch Dataset for crystal structures
"""

from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import json

class CrystalDataset(Dataset):
    def __init__(self, folder):
        print(f"Loading CrystalDataset from {folder}")

        self.data = []
        self.labels = []
        self.label_map = {"amorphous": 0, "unknown": 1}

        for structure_name in tqdm(os.listdir(folder)):
            if os.path.isdir(os.path.join(folder, structure_name)):
                for f in tqdm(os.listdir(os.path.join(folder, structure_name))):
                    if f.endswith(".npy"):
                        data = np.load(os.path.join(folder, structure_name, f))

                        # Check data integrity
                        if np.any(np.isnan(data)):
                            print(f"Skipping {f} due to NaN values")
                            continue

                        # Name should be in the form <strcture>_number.npy
                        label = structure_name
                        if label not in self.label_map:
                            self.label_map[label] = len(self.label_map)

                        self.data.append(data)
                        self.labels += [
                            self.label_map[label] for _ in range(data.shape[0])
                        ]

        self.data = np.vstack(self.data)
        self.labels = np.array(self.labels)

        # Normalization parameters (normalize inside the model instead)
        self.means = np.mean(self.data, axis=0)
        self.stds = np.std(self.data, axis=0)

        # Save label map
        label_map_path = "ml_dataset/label_map.json"
        with open(label_map_path, "w") as f:
            json.dump(self.label_map, f)

        print(
            f"Loaded dataset with {len(self.label_map)} classes and {len(self.data)} samples"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]