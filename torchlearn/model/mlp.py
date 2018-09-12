import os
from typing import *

import torch
from torch import nn

from torchlearn.utils import default_device


class MLP(nn.Module):
    """Multi-layer perceptron model"""

    def __init__(self, input_dim: int, hidden_dims: List[int]=None, output_dim: int=2):


class LogisticRegression(nn.Module):
    """Logistic Regression model"""

    def __init__(self, input_dim: int, device: str=None):
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.device = default_device() if device is None else device
        # Add bias weight
        self.weights = nn.Linear(in_features=1 + input_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

        if self.device == 'cuda':
            self.weights = self.weights.cuda()

    def forward(self, x: torch.Tensor) -> float:
        """Estimate probability of x"""
        return self.sigmoid(self.weights(x))

    def save(self, filepath: os.PathLike):
        """Save model into binary format"""
        torch.save(obj=self, f=filepath)

    @staticmethod
    def load(filepath: os.PathLike, device: str=None):
        """Load model from binary format"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.load(f=filepath, map_location=device)
