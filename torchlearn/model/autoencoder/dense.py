from typing import *

import torch
import torch.nn as nn

from .autoencoder import Autoencoder


class Dense(nn.Module):
    """MLP model"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, device: str='cpu'):
        super(Dense, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.device = device
        layers = nn.ModuleList()
        current_dim = input_dim
        for next_dim in hidden_dims:
            layers.append(nn.Linear(in_features=current_dim, out_features=next_dim))
            current_dim = next_dim
        layers.append(nn.Linear(in_features=current_dim, out_features=output_dim))
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DenseAutoencoder(Autoencoder):
    """
    Dense Autoencoder,
        takes vectorized text as input,
        encodes in into latent space,
        decodes from latent space into output space of same dimensionality as input space.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int, device: str='cpu'):
        encoder = Dense(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=latent_dim)
        decoder = Dense(input_dim=latent_dim, hidden_dims=hidden_dims[::-1], output_dim=input_dim)
        super(DenseAutoencoder, self).__init__(encoder=encoder, decoder=decoder, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, encode and decode"""
        return self.decoder(self.encoder(x))
