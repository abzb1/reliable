import torch
import torch.nn as nn

class MLP300K(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 377),
            nn.ReLU(),
            nn.Linear(377, 10)
        )
    
    def forward(self, x):
        return self.model(x)