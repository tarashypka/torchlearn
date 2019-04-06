import os
from typing import *

import torch
from torch import nn

from torchlearn.utils import default_device


class MLP(nn.Module):
    """Multi-layer perceptron model"""

    def __init__(self, input_dim: int, hidden_dims: List[int]=None, output_dim: int=1, device: str=default_device()):
        super(MLP, self).__init__()
        self.inp_dim = input_dim
        self.hidden_dims = hidden_dims
        self.outp_dim = output_dim
        self.device = device

        if not hidden_dims:
            raise AttributeError(f'Invalid value of hidden_dims = {hidden_dims}, must contain at least one dimension!')

        self.inp_layer = nn.Linear(in_features=input_dim, out_features=hidden_dims[0])

        self.hidden_layers_ = nn.ModuleList()
        for inp_dim, outp_dim in zip(self.hidden_dims[:-1], self.hidden_dims[1:]):
            layer = nn.Linear(in_features=inp_dim, out_features=outp_dim)
            nn.init.xavier_uniform_(layer.weight)
            self.hidden_layers_.append(layer)

        self.outp_layer = nn.Linear(in_features=hidden_dims[-1], out_features=output_dim)

        self.dropout_ = nn.ModuleList()
        self.batch_norm_ = nn.ModuleList()
        self.batch_norm_.append(nn.BatchNorm1d(num_features=self.inp_dim))
        self.batch_norm_.append(nn.BatchNorm1d(num_features=self.inp_layer.out_features))
        for layer in self.hidden_layers_:
            self.dropout_.append(nn.Dropout(p=0.2))
            self.batch_norm_.append(nn.BatchNorm1d(num_features=layer.out_features))

        self.activation_ = nn.ReLU()
        self.prediction_ = nn.Softmax() if output_dim > 1 else nn.Sigmoid()

        if self.device == 'cuda':
            self.layers_ = self.layers_.cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate probability of x"""
        x = self.batch_norm_[0](x)
        x = self.inp_layer(x)
        x = self.batch_norm_[1](x)
        for hidden, dropout, batch_norm in zip(self.hidden_layers_, self.dropout_, self.batch_norm_[2:]):
            x = batch_norm(dropout(self.activation_((hidden(x)))))
        y = self.prediction_(self.outp_layer(x))
        return y


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
