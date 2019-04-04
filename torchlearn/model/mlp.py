import os
from typing import *

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

from torchlearn.utils import default_device


class MLP(nn.Module):
    """Multi-layer perceptron model"""

    def __init__(self, input_dim: int, hidden_dims: List[int]=None, output_dim: int=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.layers = nn.ModuleList()
        for input_dim, output_dim in [input_dim] + hidden_dims + [output_dim]:
            self.layers.append(nn.Linear(in_features=1 + input_dim, out_features=output_dim))
        self.sigmoid = nn.Sigmoid()

        if self.device == 'cuda':
            self.input_layer = self.input_layer.cuda()
            self.hidden_layers = self.hidden_layers.cuda()

    def forward(self, x: torch.Tensor) -> np.array:
        """Estimate probability of x"""
        y: Variable = None
        for layer in self.layers:
            y = self.sigmoid(layer(x))
        return y.data.numpy()


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

    def forward(self, x: torch.Tensor) -> np.array:
        """Estimate probability of x"""
        return self.sigmoid(self.weights(x)).data.numpy()

    def save(self, filepath: os.PathLike):
        """Save model into binary format"""
        torch.save(obj=self, f=filepath)

    @staticmethod
    def load(filepath: os.PathLike, device: str=None):
        """Load model from binary format"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.load(f=filepath, map_location=device)
