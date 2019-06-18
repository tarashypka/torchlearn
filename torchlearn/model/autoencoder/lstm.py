import torch
import torch.nn as nn

from ..lstm import Lstm


class LstmAutoencoder(nn.Module):
    """
    Lstm Autoencoder,
        takes text represented as sequence of word embeddings as input,
        encodes them into latent space,
        decodes from latent space into output space of same dimensionality as input space.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super(LstmAutoencoder, self).__init__()
        self.encoder = Lstm(input_dim=input_dim, hidden_dim=latent_dim, device=self.device)
        self.decoder = Lstm(input_dim=latent_dim, hidden_dim=input_dim, device=self.device)
        if self.device == 'cuda':
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, encode and decode"""
        return self.decoder(self.encoder(x))
