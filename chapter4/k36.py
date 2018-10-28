import pickle
from collections import Counter

if __name__ == '__main__':
    with open('../../../data/neko.list', 'rb') as f:
        lines = pickle.load(f)
    words = [word['surface'] for line in lines for word in line]
    print('\n'.join(['{}    {}'.format(w,cnt) for w, cnt in Counter(words).most_common()]))
    with open('../../../data/neko.vocab', 'wb') as f:
        pickle.dump(Counter(words),f)
