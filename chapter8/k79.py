from sklearn.metrics import precision_recall_curve
from k76 import labeling
import matplotlib.pyplot as plt


def draw_figure(X, Y):
    plt.plot(X, Y)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('../result/k79.png')

def main():
    trues, _, probs = labeling()
    precision, recall, thr = precision_recall_curve(y_true=trues, probas_pred=[p[0] for p in probs])
    draw_figure(recall, precision)


if __name__ == '__main__':
    main()