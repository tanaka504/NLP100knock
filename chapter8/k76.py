from k73 import train
import numpy as np


def labeling():
    model, vocab = train()
    X_test = np.array([np.sum([np.eye(vocab.vocab_size+1)[wid] for wid in sentence], axis=0) for sentence in vocab.sentences])
    true_label = [label[0] for label in vocab.labels]
    pred_label = model.predict(X_test)
    pred_prob = model.predict_proba(X_test)
    print('true|pred|prob')
    [print('{}\t{}\t{}'.format(true, pred, max(prob))) for true, pred, prob in zip(true_label, pred_label, pred_prob)]
    return true_label, pred_label, pred_prob

if __name__ == '__main__':
    labeling()