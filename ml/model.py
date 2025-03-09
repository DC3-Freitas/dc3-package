import torch
import torch.nn as nn

class MLP_Model(nn.Module):
    def __init__(self):
        super(MLP_Model, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(330, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 6),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):

        return self.network(x)