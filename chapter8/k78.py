from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from k73 import train
import numpy as np

def cross_validation():
    _, vocab = train()
    X_tensor = np.array([np.sum([np.eye(vocab.vocab_size+1)[wid] for wid in sentence], axis=0) for sentence in vocab.sentences])
    Y_tensor = np.array(vocab.labels)
    preds = cross_val_predict(LogisticRegression(), X_tensor, Y_tensor, cv=5)
    print('Precision: ', precision_score(vocab.labels, preds))
    print('Recall: ', recall_score(vocab.labels, preds))
    print('F-measure: ', f1_score(vocab.labels, preds))


if __name__ == '__main__':
    cross_validation()