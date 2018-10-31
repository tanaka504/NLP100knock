from k76 import labeling
from sklearn.metrics import f1_score, recall_score, precision_score


def evaluation():
    true, pred, _ = labeling()
    print('Precision: ', precision_score(y_true=true, y_pred=pred))
    print('Recall: ', recall_score(y_true=true, y_pred=pred))
    print('F-measure: ', f1_score(y_true=true, y_pred=pred))


if __name__ == '__main__':
    evaluation()