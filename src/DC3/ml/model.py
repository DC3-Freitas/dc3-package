"""
MLP model architecture for crystal structure classification with
built-in feature normalization.
"""

import numpy as np
import torch
from torch import nn
import warnings


class MLPModel(nn.Module):
    """
    Multilayer perceptron (MLP) model for structural classification.
    Also handles input normalization interally (so does not expect normalized data).
    """

    def __init__(
        self,
        classes: int,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
    ) -> None:
        """
        Initialize the MLP model.

        Args:
            classes: number of output classes.
            means: mean values for input normalization.
            stds: standard deviation values for input normalization.
        """
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

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Performs forward pass in the NN.

        Args:
            x: input tensor you want to perform ineference on
        Returns:
            Output of the model after passing in x
        """
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
