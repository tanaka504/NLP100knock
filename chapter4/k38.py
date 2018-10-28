import pickle
from collections import Counter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('../../../data/neko.vocab', 'rb') as f:
        w_cnt = pickle.load(f)
    counts = [cnt for w, cnt in w_cnt.most_common()]

    plt.hist(counts,bins=100,range=(2,100))
    plt.show()