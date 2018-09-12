from pathlib import Path

from torchlearn.callback import SaveCallback
from torchlearn.vectorizer import TextTokenizer, EmbeddingTextVectorizer
from torchlearn.model.autoencoder import LstmAutoencoder
from torchlearn.trainer.autoencoder import LstmAutoencoderTrainer
from torchlearn.utils import load_pickle, read_lines, report


EMBEDDINGS_PATH = Path('/home/tas/data/embeddings.pickle')
VOCAB_PATH = Path('/home/tas/data/vocab.txt')
TOKENIZER_PATH = Path('/home/tas/data/tokenizer.bin')
VECTORIZER_PATH = Path('/home/tas/data/vectorizer.bin')
TEXTS_PATHS = [Path(f'/home/tas/data/texts/texts_{i}.txt') for i in range(100)]
report('There are', len(TEXTS_PATHS), 'texts files')

vocab = list(read_lines(filepath=VOCAB_PATH))
report('There are', len(vocab), 'types in vocab')
embeddings = load_pickle(filepath=EMBEDDINGS_PATH)
report('There are', embeddings.shape[0], 'embeddings')
tokenizer = TextTokenizer.__load__(filepath=TOKENIZER_PATH)
vectorizer = EmbeddingTextVectorizer(
    types=vocab,
    embeddings=embeddings,
    tokenizer=tokenizer
)
vectorizer.__save__(filepath=VECTORIZER_PATH)
lstm = LstmAutoencoder(
    input_dim=embeddings.shape[1],
    latent_dim=256
)
trainer = LstmAutoencoderTrainer(
    autoencoder=lstm,
    vectorizer=vectorizer,
    texts_paths=TEXTS_PATHS,
    texts_in_file=4096,
    batch_size=64,
    verbosity=1
)
trainer.add_callback(SaveCallback(obj_to_save=trainer, save_per_min=1, save_name='lstm_trainer'))
trainer.add_callback(SaveCallback(obj_to_save=lstm, save_per_min=1, save_name='lstm_model'))
trainer.add_callback(SaveCallback(obj_to_save=lstm.encoder, save_per_min=1, save_name='lstm_encoder'))
trainer.train(n_epochs=10, progress_bar=None)
