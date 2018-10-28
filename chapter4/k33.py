import pickle

if __name__ == '__main__':
    with open('../../../data/neko.list', 'rb') as f:
        lines = pickle.load(f)
    print('\n'.join([word['surface'] for line in lines for word in line if word['pos'] == '名詞' and word['pos2'] == 'サ変接続']))