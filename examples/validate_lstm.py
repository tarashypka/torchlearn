from pathlib import Path

import torch

from torchlearn.vectorizer import EmbeddingTextVectorizer
from torchlearn.utils import report, read_lines


TEXTS_PATH = Path(f'/home/tas/data/texts/texts_0.txt')
VECTORIZER_PATH = Path('/home/tas/data/vectorizer.bin')
ENCODER_PATH = Path('/home/tas/data/encoder.bin')

vectorizer: EmbeddingTextVectorizer = EmbeddingTextVectorizer.__load__(filepath=VECTORIZER_PATH)
encoder = torch.load(f=ENCODER_PATH)

texts = list(read_lines(filepath=TEXTS_PATH))[:16]
report('Vectorize', len(texts), 'texts ...')
vectors = vectorizer.transform(texts=texts)
report(vectors.shape)
report('Encode', vectors.shape[0], 'vectors ...')
embeddings = encoder(vectors)
report(embeddings.shape)
