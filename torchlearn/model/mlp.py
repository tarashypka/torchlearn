import os
from typing import *

import torch
from torch import nn

from torchlearn.utils import default_device


class MLP(nn.Module):
    """Multi-layer perceptron model"""

    def __init__(self, input_dim: int, hidden_dims: List[int]=None, output_dim: int=2, device: str=default_device()):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.device = device

        self.layers_ = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            layer = nn.Linear(in_features=input_dim, out_features=output_dim)
            nn.init.xavier_uniform_(layer.weight)
            self.layers_.append(layer)
        self.activation_ = nn.ReLU()
        self.prediction_ = nn.Sigmoid()

        if self.device == 'cuda':
            self.layers_ = self.layers_.cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate probability of x"""
        hidden_layers = self.layers_[:-1]
        output_layer = self.layers_[-1]
        for layer in hidden_layers:
            x = self.activation_(layer(x))
        return self.prediction_(output_layer(x))


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

    def save(self, filepath: os.PathLike):
        """Save model into binary format"""
        torch.save(obj=self, f=filepath)

    @staticmethod
    def load(filepath: os.PathLike, device: str=None):
        """Load model from binary format"""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.load(f=filepath, map_location=device)
