
#### fun with pytorch


##### Install
```
$ cd torchlearn && pip install .
```

###### TextVectorizer
```
from torchtext.vectorizer import TextVectorizer
types = ['a', 'b', 'c']
embeddings = np.random.random(size=(len(types), 5))
texts = ['a b', 'c d']
vectors = TextVectorizer(types=types, embeddings=embeddings, seq_len=4).transform(texts)
print(vectors.shape)
(4, 2, 5)
```
