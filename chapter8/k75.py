from k73 import Vocab, train
import numpy as np


def main():
    _, vocab = train()
    X_tensor = np.sum([np.sum([np.eye(vocab.vocab_size+1)[wid] for wid in sentence], axis=0) for sentence in vocab.sentences], axis=0)
    [print(vocab.id2word[idx]) for idx in X_tensor.argsort()[::-1][:10]]
    [print(vocab.id2word[idx]) for idx in X_tensor.argsort()[:10]]


if __name__ == '__main__':
    main()