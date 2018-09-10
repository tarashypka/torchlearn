import os
from collections import Counter
from typing import *

import numpy as np
import torch
from torch.autograd import Variable
from torchtext.data import Field
from torchtext.vocab import Vocab
from torchlearn.utils import report, dump_pickle, load_pickle


class TextVectorizer:
    """Text vectorizer on top of torchtext"""

    __UNKNOWN__ = '<unk>'
    __PADDING__ = '<pad>'

    def __init__(self, types: List[str], embeddings: np.array, seq_len: int=128, device: str=None):
        self.seq_len = seq_len
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        special_types = [TextVectorizer.__UNKNOWN__, TextVectorizer.__PADDING__]
        self.types_ = special_types + types
        # Initialize embeddings for special types with random values
        special_embeddings = np.random.random(size=(len(special_types), embeddings.shape[1]))
        embeddings = np.concatenate([special_embeddings, embeddings])
        embeddings = Variable(torch.Tensor(embeddings, device=device), requires_grad=False)
        self.embeddings_ = embeddings
        self.text_field_ = Field(
            fix_length=seq_len, pad_token=TextVectorizer.__PADDING__, pad_first=True)
        # Create vocab from fake counts, reverse types for torchtext to preserve their order
        freqs = Counter({t: i for i, t in enumerate(special_types + types[::-1])})
        self.text_field_.vocab = Vocab(counter=freqs, specials=special_types)

    def tokenize_(self, text: str) -> List[str]:
        tokens = text.split()
        return tokens

    def transform_(self, texts: List[str]) -> torch.Tensor:
        texts = [self.tokenize_(text) for text in texts]
        texts = self.text_field_.process(batch=texts)
        return self.embeddings_[texts]

    def transform(self, texts: List[str]) -> torch.Tensor:
        """Transform batch of texts into (seq_len, batch_size, dim) tensor of embeddings"""
        return self.transform_(texts=texts)

    def save(self, filepath: os.PathLike):
        """Save vectorizer into binary format"""
        report('Save vectorizer into', filepath, '...')
        dump_pickle(filepath=filepath, obj=self)

    @staticmethod
    def load(filepath: os.PathLike):
        """Load vectorizer from binary format"""
        report('Load vectorizer from', filepath, '...')
        return load_pickle(filepath=filepath)
