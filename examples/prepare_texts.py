#!/usr/local/anaconda/bin/python

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from torchlearn.vectorizer import TextTokenizer
from torchlearn.utils import write_lines, read_lines, clear_dir
from tqdm import tqdm


SQLITE_PATH = Path(f'/home/tas/data/sqlite/ru.db')
STOPTERMS_PATH = Path(f'/home/tas/data/stopterms.txt')
TEXTS_DIR = Path('/home/tas/data/texts')
clear_dir(TEXTS_DIR)
TOKENIZER_PATH = Path('/home/tas/data/tokenizer.bin')
BATCH_SIZE = 4096

STOPTERMS = list(read_lines(filepath=STOPTERMS_PATH))
tokenizer = TextTokenizer(stopterms=STOPTERMS)
sqlite = sqlite3.connect(str(SQLITE_PATH))
qry = f"""
SELECT id, text_searchable
FROM job_text
WHERE id >= {{first_id}}
ORDER BY id ASC
LIMIT {BATCH_SIZE};"""
n_texts = pd.read_sql(sql="SELECT count(*) AS count FROM job_text;", con=sqlite).loc[0, 'count']
n_batches = np.ceil(n_texts / BATCH_SIZE).astype(int)
first_id = 0
for b in tqdm(range(n_batches)):
    texts = pd.read_sql(sql=qry.format(first_id=first_id), con=sqlite)
    first_id = texts['id'].max() + 1
    texts['text_searchable'] = texts['text_searchable'].apply(tokenizer.tokenize).str.join(sep=' ')
    texts = texts['text_searchable'].tolist()
    write_lines(filepath=TEXTS_DIR / f'texts_{b}.txt', lines=texts)

tokenizer.__save__(filepath=TOKENIZER_PATH)
