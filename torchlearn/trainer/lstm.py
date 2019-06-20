from typing import *

import torch
from torch import nn
from torch import optim

from .trainer import Trainer
from torchlearn.model import Lstm


class LstmTrainer(Trainer):

    def __init__(self, model: Lstm, loss=None, optimizer=None, workers: int=1):
        super(LstmTrainer, self).__init__()
        self.model = model
        if loss is None:
            loss = nn.MSELoss(reduction='mean')
        self.loss = loss
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.optimizer = optimizer
        self.workers = workers

        torch.set_num_threads(self.workers)

    def train(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        self.call_callbacks()
        return y_pred, loss.item()
