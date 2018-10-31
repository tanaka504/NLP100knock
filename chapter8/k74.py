from k73 import Vocab, train
import numpy as np

def predict():
    model, vocab = train()
    X_test = np.array([np.sum([np.eye(vocab.vocab_size+1)[wid] for wid in sentence], axis=0) for sentence in vocab.sentences])
    print(model.predict(X_test))

if __name__ == '__main__':
    predict()