from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from k73 import train


def cross_validation():
    _, vocab = train()
    preds = cross_val_predict(LogisticRegression(), vocab.sentences, vocab.labels, cv=5)
    print('Precision: ', precision_score(vocab.labels, preds))
    print('Recall: ', recall_score(vocab.labels, preds))
    print('F-measure: ', f1_score(vocab.labels, preds))


if __name__ == '__main__':
    cross_validation()