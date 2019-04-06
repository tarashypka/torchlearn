from pathlib import Path

import torch
from torch import nn

from torchlearn.utils import default_device


class LogisticRegression(nn.Module):
    """Logistic Regression model"""

    def __init__(self, input_dim: int, device: str=default_device()):
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.device = device

        # Add bias weight
        self.weights_ = nn.Linear(in_features=1 + input_dim, out_features=1)
        self.sigmoid_ = nn.Sigmoid()

        if self.device == 'cuda':
            self.weights_ = self.weights_.cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate probability of x"""
        return self.sigmoid_(self.weights_(x))

    def save(self, filepath: Path):
        """Save model into binary format"""
        torch.save(obj=self, f=filepath)

    @staticmethod
    def load(filepath: Path, device: str=None):
        """Load model from binary format"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.load(f=filepath, map_location=device)
