from torch.utils.data import Dataset
import numpy as np
import json
import os


class CrystalDataset(Dataset):
    """
    Dataset for crystal structure classification. Mean and stds are computed
    for reference but the dataset itself does not normalize.

    Attributes:
        features: array containing all feature vectors.
        labels: array of shape containing all integer class labels.
        label_map: mapping from structure name to corresponding integer class label.
        means: per-feature means computed across the dataset.
        stds: per-feature standard deviations computed across the dataset.
    """

    def __init__(
        self, data: list[tuple[str, np.ndarray]], save_label_map_dir: str | None = None
    ) -> None:
        """
        Initializes the dataset from a list of (structure name, feature matrix) pairs.

        Args:
            data: list of structure names and associated feature arrays of shape
            save_label_map_dir: if provided, saves the structure-to-label mapping to this directory as a JSON file.
        """
        # Create label and label map
        self.labels = []
        self.label_map = {}

        for structure, features in data:
            assert not np.any(
                np.isnan(features)
            ), f"Found nan in features associated with {structure}"

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
        """Returns the number of feature vectors in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Returns the (feature, label) pair at the specified index."""
        return self.features[idx], self.labels[idx]
