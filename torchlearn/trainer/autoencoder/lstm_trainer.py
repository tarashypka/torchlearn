import os
import random
from typing import *

from torch import nn
from torch import optim
from tqdm import tqdm

from torchlearn.utils import read_lines, report, batches, plain_path
from torchlearn.vectorizer import EmbeddingTextVectorizer
from torchlearn.model.autoencoder import LstmAutoencoder
from torchlearn.trainer import Trainer


class LstmAutoencoderTrainer(Trainer):
    """Learner to train LstmAutoencoder"""

    def __init__(
            self,
            autoencoder: LstmAutoencoder,
            vectorizer: EmbeddingTextVectorizer,
            texts_paths: List[os.PathLike],
            texts_in_file: int,
            loss=None,
            optimizer=None,
            batch_size: int=64,
            verbosity: int=1):

        self.autoencoder = autoencoder
        self.vectorizer = vectorizer
        self.texts_paths = [plain_path(path) for path in texts_paths]
        self.texts_in_file = texts_in_file
        if loss is None:
            loss = nn.MSELoss(reduction='elementwise_mean')
        self.loss = loss
        if optimizer is None:
            optimizer = optim.SGD(self.autoencoder.parameters(), lr=0.1)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbosity = verbosity

        self.epoch_: int = 0
        self.iter_: int = 0

    def report(self, *args, **kwargs):
        if self.verbosity > 0:
            report(*args, **kwargs)

    def train(self, n_epochs: int, progress_bar=tqdm):
        n_batches = len(self.texts_paths) * self.texts_in_file // self.batch_size
        for epoch in range(n_epochs):
            self.report('Run epoch', epoch, '...')
            self.iter_ = 0
            batch = 0
            error = 0
            samples = 0
            random.shuffle(self.texts_paths)
            for texts_path in progress_bar(self.texts_paths):
                texts = list(read_lines(filepath=texts_path))
                random.shuffle(texts)
                for batch_texts in batches(texts, size=self.batch_size):
                    x = self.vectorizer.transform(texts=batch_texts)
                    y = self.autoencoder(x)
                    loss = self.loss(y, x)
                    error += loss.data.tolist() * len(batch_texts)
                    samples += len(batch_texts)
                    loss.backward()
                    self.optimizer.step()
                    batch += 1
                    self.report('{batch}/{batches} Error={error:.4f}'.format(
                        batch=batch, batches=n_batches, error=error / samples))
                    self.iter_ += 1
            self.epoch_ += 1
