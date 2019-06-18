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


class LstmClassifier(Lstm):
    """Lstm model with linear layer on top"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: str=default_device()):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, device=device)
        self.output_dim = output_dim
        self.hidden2label = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x=x)
        return self.hidden2label(y[-1])
