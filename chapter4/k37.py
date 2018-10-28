import pickle
from collections import Counter
import matplotlib.pyplot as plt


if __name__ == '__main__':
    with open('../../../data/neko.vocab', 'rb') as f:
        w_cnt = pickle.load(f)
    print(w_cnt.most_common(10))

    plt.bar(left=[_ for _ in range(10)],
            height=[cnt for word, cnt in w_cnt.most_common(10)])
    plt.xticks(range(10), [word for word, cnt in w_cnt.most_common(10)])
    plt.show()