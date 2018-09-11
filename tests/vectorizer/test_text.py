import unittest

import numpy as np
import torch
from torchlearn.vectorizer import TextTokenizer, EmbeddingTextVectorizer


class TextTokenizerTest(unittest.TestCase):

    def test_tokenize_pattern(self):
        """Test token_pattern tokenizer parameter"""
        tokenizer = TextTokenizer(token_pattern='^[^\W\d][^\W\d]+$', max_token_len=8)
        text = 'a bc def a1 bc2 def3'
        tokens = tokenizer.tokenize(text)
        for token in ['a', 'a1', 'bc2', 'def3']:
            self.assertNotIn(member=token, container=tokens, msg="Invalid token was not removed!")
        for token in ['bc', 'def']:
            self.assertIn(member=token, container=tokens, msg="Valid token was removed!")

    def test_tokenize_token_len(self):
        """Test mak_token_len tokenizer parameter"""
        tokenizer = TextTokenizer(token_pattern='^[^\W\d][^\W\d]+$', max_token_len=4)
        text = 'abc abcd abcde'
        tokens = tokenizer.tokenize(text)
        self.assertIn(member='abc', container=tokens, msg="Short enough token was removed!")
        self.assertIn(member='abcd', container=tokens, msg="Short enough token was removed!")
        self.assertNotIn(member='abcde', container=tokens, msg="Too long token was not removed!")

    def test_tokenize_stopwords(self):
        """Test stopwords tokenizer parameter"""
        tokenizer = TextTokenizer(token_pattern='^[^\W\d][^\W\d]+$', max_token_len=8, stopterms=['abc'])
        text = 'abc def gjk'
        tokens = tokenizer.tokenize(text)
        self.assertNotIn(member='abc', container=tokens, msg="Stopword was not removed!")
        self.assertIn(member='def', container=tokens, msg="Not a stopword was removed!")
        self.assertIn(member='gjk', container=tokens, msg="Not a stopword was removed!")


class EmbeddingTextVectorizerTest(unittest.TestCase):

    def test_transform(self):
        """Test that transform() method returns proper vectors"""
        types = ['abc', 'def', 'gjk', 'lmn']
        embeddings = np.random.random(size=(len(types), 200))
        tokenizer = TextTokenizer()
        vectorizer = EmbeddingTextVectorizer(types=types, embeddings=embeddings, tokenizer=tokenizer, seq_len=8)
        texts = ['abc def gjk', 'unknown gjk']
        vectors = vectorizer.transform(texts=texts)

        self.assertEqual(
            vectors.shape[0], vectorizer.seq_len, "First dimension should be sequence length!")
        self.assertEqual(
            vectors.shape[1], len(texts), "Second dimension should be amount of input texts!")
        self.assertEqual(
            vectors.shape[2], embeddings.shape[1], "Third dimension should be embedding dimension!")

        token_to_emb = dict(zip(vectorizer.types_, vectorizer.embeddings_))
        pad_true = token_to_emb[vectorizer.__PADDING__]
        unk_true = token_to_emb[vectorizer.__UNKNOWN__]
        for text_ind, text in enumerate(texts):
            tokens = tokenizer.tokenize(text)
            text_len = len(tokens)
            n_paddings = vectorizer.seq_len - text_len
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
