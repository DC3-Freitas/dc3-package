import torch
import torch.nn as nn

class MLP_Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.network = nn.sequential(
            nn.Linear(330, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.fc(x)