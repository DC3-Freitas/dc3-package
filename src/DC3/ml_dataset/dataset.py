"""
PyTorch Dataset for crystal structures
"""

from torch.utils.data import Dataset
import numpy as np
import json
import os

class CrystalDataset(Dataset):
    def __init__(self, data: list[tuple[str, np.ndarray]], save_label_map_dir: None | str = None):
        # Create label and label map
        self.labels = []
        self.label_map = {}

        for structure, features in data:
            assert not np.any(np.isnan(features)), f"Found nan in features associated with {structure}"

            if structure not in self.label_map:
                self.label_map[structure] = len(self.label_map)

            self.labels += [self.label_map[structure] for _ in range(features.shape[0])]

        # Get features and make labels into numpy form
        self.features = np.vstack([features.copy() for _, features in data])
        self.labels = np.array(self.labels)

        # Normalization parameters (normalize inside the model instead)
        self.means = np.mean(self.features, axis=0)
        self.stds = np.std(self.features, axis=0)

        # Save label map if requested
        if save_label_map_dir is not None:
            with open(os.path.join(save_label_map_dir, "label_map.json"), "w") as f:
                json.dump(self.label_map, f)

        print(
            f"\nLoaded dataset with {len(self.label_map)} classes and {len(self.features)} samples"
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
