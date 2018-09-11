import os
from torch import nn

import torch


class Autoencoder(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: str=None):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        if device == 'cuda':
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def save(self, filepath: os.PathLike):
        """Save autoencoder into binary format"""
        torch.save(obj=self, f=filepath)

    @staticmethod
    def load(filepath: os.PathLike, map_location):
        """Load autoencoder from binary format"""
        return torch.load(f=filepath, map_location=map_location)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass