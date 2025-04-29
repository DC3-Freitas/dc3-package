import torch
import torch.nn as nn
import warnings


class MLPModel(nn.Module):
    def __init__(self, classes, means=None, stds=None):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(330, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, classes),
            nn.LogSoftmax(dim=1),
        )

        # If no mean or std is provided put dummy values which **should be overriden before the model is used**
        means = (
            torch.zeros((1, 330))
            if means is None
            else torch.tensor(means, dtype=torch.float32).unsqueeze(0)
        )
        stds = (
            torch.ones((1, 330))
            if stds is None
            else torch.tensor(stds, dtype=torch.float32).unsqueeze(0)
        )

        self.register_buffer("means", means)
        self.register_buffer("stds", stds)

    def forward(self, x):
        # Warn if we attempted to do forward prop but the means or stds still match their default values
        if torch.equal(self.means, torch.zeros_like(self.means)):
            warnings.warn(
                "Value of 'means' is still default value. Double check if you actually set means"
            )
        if torch.equal(self.stds, torch.ones_like(self.stds)):
            warnings.warn(
                "Value of 'stds' is still default value. Double check if you actually set stds"
            )

        normalized_x = (x - self.means) / (self.stds + 1e-6)
        return self.network(normalized_x)
