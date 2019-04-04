import os
from typing import *

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

from torchlearn.utils import default_device


class MLP(nn.Module):
    """Multi-layer perceptron model"""

    def __init__(self, input_dim: int, hidden_dims: List[int]=None, output_dim: int=1, device: str=default_device()):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.device = device

        self.layers_ = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        for input_dim, output_dim in zip(dims[:-1], dims[1:]):
            self.layers_.append(nn.Linear(in_features=1 + input_dim, out_features=output_dim))
        self.activation_ = nn.ReLU()
        self.sigmoid_ = nn.Sigmoid()

        if self.device == 'cuda':
            self.layers_ = self.layers_.cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate probability of x"""
        for layer in self.layers_:
            bias = torch.ones(size=(x.shape[0], 1))
            x = torch.cat((x, bias), dim=1)
            x = self.activation_(layer(x))
        return self.sigmoid_(x)


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
