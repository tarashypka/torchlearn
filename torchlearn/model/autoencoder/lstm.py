import torch
import torch.nn as nn
from torch.autograd import Variable

from torchlearn.utils import default_device


class Lstm(nn.Module):
    """LSTM model"""

    def __init__(self, input_dim: int, hidden_dim: int, device: str=default_device()):
        super(Lstm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)

    def get_hidden(self, batch_size: int):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim, device=self.device))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim, device=self.device))
        return h0, c0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x, self.get_hidden(batch_size=x.shape[1]))
        return y


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
