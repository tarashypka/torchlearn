import regex as re
from collections import Counter
from typing import *

import numpy as np
import torch
from torch.autograd import Variable
from torchtext.data import Field
from torchtext.vocab import Vocab
from torchlearn.utils import Savable, Loadable
from sklearn.feature_extraction.text import TfidfVectorizer


class TextTokenizer(Savable, Loadable):

    def __init__(
            self, token_pattern: str='^[^\W\d][^\W\d]+$', max_token_len: int=32,
            stopterms: List[str]=None):

        self.token_pattern = token_pattern
        self.max_token_len = max_token_len
        self.stopterms = [] if stopterms is None else stopterms

    def is_valid(self, word: str) -> bool:
        return re.match(self.token_pattern, word) and len(word) <= self.max_token_len

    @staticmethod
    def tokenize_word(word: str) -> str:
        return word.lower()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return [self.tokenize_word(word) for word in text.split()
                if self.is_valid(word) and word not in self.stopterms]


class TextVectorizer(Savable, Loadable):

    def transform(self, texts: List[str]) -> torch.Tensor:
        pass


class EmbeddingTextVectorizer(TextVectorizer):
    """Word embeddings text vectorizer on top of torchtext"""

    __UNKNOWN__ = '<unk>'
    __PADDING__ = '<pad>'

    def __init__(
            self, types: List[str], embeddings: np.array, tokenizer: TextTokenizer,
            seq_len: int=128, device: str=None):

        self.seq_len = seq_len
        self.tokenizer = tokenizer
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        special_types = [EmbeddingTextVectorizer.__UNKNOWN__, EmbeddingTextVectorizer.__PADDING__]
        self.types_ = special_types + types
        # Initialize embeddings for special types with random values
        special_embeddings = np.random.random(size=(len(special_types), embeddings.shape[1]))
        embeddings = np.concatenate([special_embeddings, embeddings])
        embeddings = Variable(torch.Tensor(embeddings, device=device), requires_grad=False)
        self.embeddings_ = embeddings
        self.text_field_ = Field(
            fix_length=seq_len, pad_token=EmbeddingTextVectorizer.__PADDING__, pad_first=True)
        # Create vocab from fake counts, reverse types for torchtext to preserve their order
        freqs = Counter({t: i for i, t in enumerate(special_types + types[::-1])})
        self.text_field_.vocab = Vocab(counter=freqs, specials=special_types)

    def transform(self, texts: List[str]) -> torch.Tensor:
        """Transform batch of texts into (seq_len, batch_size, dim) tensor of embeddings"""
        texts = [self.tokenizer.tokenize(text) for text in texts]
        texts = self.text_field_.process(batch=texts)
        return self.embeddings_[texts]


class TfidfTextVectorizer(TextVectorizer):
    """Tfidf vectors text vectorizer"""

    def __init__(self, tfidf: TfidfVectorizer):
        self.tfidf = tfidf

    def transform(self, texts: List[str]) -> torch.Tensor:
        vectors = self.tfidf.transform(texts)
        return vectors.dense()
