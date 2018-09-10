import unittest

import numpy as np
import torch
from torchlearn.vectorizer import TextVectorizer


class TextVectorizerTest(unittest.TestCase):

    def test_transform(self):
        """Test that transform() method returns proper vectors"""
        types = ['a', 'b', 'c', 'd']
        embeddings = np.random.random(size=(len(types), 200))
        v = TextVectorizer(types=types, embeddings=embeddings, seq_len=8)
        texts = ['a b d', 'unknown d']
        vectors = v.transform(texts=texts)

        self.assertEqual(
            vectors.shape[0], v.seq_len, "First dimension should be sequence length!")
        self.assertEqual(
            vectors.shape[1], len(texts), "Second dimension should be amount of input texts!")
        self.assertEqual(
            vectors.shape[2], embeddings.shape[1], "Third dimension should be embedding dimension!")

        token_to_emb = dict(zip(v.types_, v.embeddings_))
        pad_true = token_to_emb[v.__PADDING__]
        unk_true = token_to_emb[v.__UNKNOWN__]
        for text_ind, text in enumerate(texts):
            tokens = v.tokenize_(text)
            text_len = len(tokens)
            n_paddings = v.seq_len - text_len
            for pad_ind in range(n_paddings):
                pad_got = vectors[pad_ind, text_ind, :]
                self.assertTrue(torch.all(torch.eq(pad_got, pad_true)), "Wrong padding vector!")
            for token_ind, token in enumerate(tokens, n_paddings):
                emb_true = token_to_emb.get(token, unk_true)
                emb_got = vectors[token_ind, text_ind, :]
                self.assertTrue(
                    torch.all(torch.eq(emb_got, emb_true)),
                    f"Wrong vector for token {token} at position {token_ind}!")


if __name__ == '__main__':
    unittest.main()
