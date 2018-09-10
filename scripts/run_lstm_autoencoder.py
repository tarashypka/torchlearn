#!/usr/local/anaconda/bin/python

from pathlib import Path
import random

from torch import optim
from torch import nn

from torchlearn.utils import read_lines, report, load_pickle, batches
from torchlearn.vectorizer import TextVectorizer
from torchlearn.autoencoder import LstmAutoencoder


TEXTS_IN_FILE = 4096
BATCH_SIZE = 64

VOCAB_PATH = Path('/home/tas/data/vocab.txt')
EMBEDDINGS_PATH = Path('/home/tas/data/embeddings.pickle')
TEXTS_PATHS = list(Path('/home/tas/data/texts').iterdir())

vocab = list(read_lines(filepath=VOCAB_PATH))
embeddings = load_pickle(filepath=EMBEDDINGS_PATH)
vectorizer = TextVectorizer(types=vocab, embeddings=embeddings, seq_len=128)
autoencoder = LstmAutoencoder(input_dim=embeddings.shape[1], hidden_dim=256)
optimizer = optim.SGD(autoencoder.parameters(), lr=0.1)
loss = nn.MSELoss(reduction='elementwise_mean')

random.shuffle(TEXTS_PATHS)
n_batches = len(TEXTS_PATHS) * TEXTS_IN_FILE // BATCH_SIZE
batch = 0
se = 0
samples = 0
for texts_path in TEXTS_PATHS:
    texts = list(read_lines(filepath=texts_path))
    random.shuffle(texts)
    for batch_texts in batches(texts, size=BATCH_SIZE):
        x = vectorizer.transform(texts=batch_texts)
        y = autoencoder(x)
        mse = loss(y, x)
        se += mse.data.tolist() * len(batch_texts)
        samples += len(batch_texts)
        mse.backward()
        optimizer.step()
        batch += 1
        report('{batch}/{batches} MSE={mse:.4f}'.format(batch=batch, batches=n_batches, mse=se / samples))
