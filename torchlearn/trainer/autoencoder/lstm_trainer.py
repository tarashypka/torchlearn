import os
import random
from typing import *

from torch import nn
from torch import optim
from tqdm import tqdm

from torchlearn.utils import read_lines, report, batches, plain_path, avg_loss
from torchlearn.vectorizer import EmbeddingTextVectorizer
from torchlearn.model.autoencoder import LstmAutoencoder
from torchlearn.trainer import Trainer


class LstmAutoencoderTrainer(Trainer):
    """Trainer to learn LSTM Autoencoder model"""

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

        super(LstmAutoencoderTrainer, self).__init__()
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
        self.cache_dir_ = plain_path('.torchtext')
        self.cache_dir_.mkdir(exist_ok=True, parents=True)

    def report(self, *args, **kwargs):
        if self.verbosity > 0:
            report(*args, **kwargs)

    def train(self, n_epochs: int, progress_bar=tqdm, save_per_min: int=None):
        n_batches = len(self.texts_paths) * self.texts_in_file // self.batch_size
        for epoch in range(1, n_epochs + 1):
            self.report('Run epoch', epoch, '...')
            self.iter_ = 0
            batch = 0
            prev_loss = None
            random.shuffle(self.texts_paths)
            texts_paths = self.texts_paths
            if progress_bar is not None:
                texts_paths = progress_bar(texts_paths)
            for texts_path in texts_paths:
                texts = list(read_lines(filepath=texts_path))
                random.shuffle(texts)
                for batch_texts in batches(texts, size=self.batch_size):
                    # CPU: 0.11sec, GPU:
                    x = self.vectorizer.transform(texts=batch_texts)
                    # CPU: 0.54sec, GPU:
                    y = self.autoencoder(x)
                    # CPU: 0.01sec, GPU:
                    loss = self.loss(y, x)
                    # CPU: 2.03sec, GPU:
                    loss.backward()
                    # CPU: 0.001sec, GPU:
                    self.optimizer.step()

                    if batch % 10 == 0:
                        curr_loss = loss.data.tolist() ** 0.5
                        err = avg_loss(curr_loss=curr_loss, prev_loss=prev_loss)
                        prev_loss = curr_loss
                        #print('X', x[-1, 0, :8])
                        #print('Y', y[-1, 0, :8])
                        self.report('{batch}/{batches} Error={error:.4f}'.format(batch=batch, batches=n_batches, error=err))

                    for callback in self.callbacks:
                        callback()

                    batch += 1
                    self.iter_ += 1

            self.epoch_ += 1
