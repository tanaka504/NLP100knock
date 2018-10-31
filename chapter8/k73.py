from sklearn.linear_model import LogisticRegression
import numpy as np
import random, pickle, os
from collections import Counter

class Vocab:
    def __init__(self, filename, vocab_size=1000):
        self.filename = filename
        self.sentences = None
        self.labels = None
        self.vocab_size = vocab_size
        self.create_vocab()

    def create_vocab(self):
        with open(self.filename, 'r') as f:
            data = f.read().split('\n')
            sentences = [[word for word in line.split(' ')[1:]]for line in data]
            label = [line.split(' ')[0] for line in data]
        freq = Counter([word for sentence in sentences for word in sentence])

        self.dic = {k for (k,v) in freq.most_common()[:self.vocab_size]}
        self.id2word = {v: k for k, v in self.dic.items()}
        self.sentences = [[self.dic[word] for word in sentence if word in self.dic] for sentence in sentences]
        self.labels = label


def train():
    if os.path.exists('../../../data/polaritydata/vocab.cls'):
        vocab = load('../../../data/polaritydata/vocab.cls')
    else:
        vocab = Vocab('../../../data/polaritydata/stemmed.txt', vocab_size=1000)
        assert len(vocab.sentences) == len(vocab.labels), 'Unexpect index'
        save(vocab, '../../../data/polaritydata/vocab.cls')
    X_tensor = np.array([np.sum([np.eye(vocab.vocab_size)[wid] for wid in sentence]) for sentence in vocab.sentences])
    Y_tensor = np.array([[1, -1] if label == '+1' else [-1, 1] for label in vocab.labels])

    model = LogisticRegression()
    model.fit(X_tensor, Y_tensor)
    return model, vocab

def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

if __name__ == '__main__':
    train()