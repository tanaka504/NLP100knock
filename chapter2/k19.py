from pprint import pprint

if __name__ == '__main__':
    vocab = {}
    with open('../../../data/hightemp.txt', 'r') as f:
        data = f.read().split('\n')
        data.remove('')
        for line in data:
            cols = line.split('\t')
            chars = list(cols[0])
            for char in chars:
                if not char in vocab:
                    vocab[char] = 1
                else:
                    vocab[char] += 1
        pprint(sorted(vocab.items(), key=lambda x:-x[1]))