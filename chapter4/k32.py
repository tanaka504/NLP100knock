import pickle

if __name__ == '__main__':
    with open('../../../data/neko.list', 'rb') as f:
        lines = pickle.load(f)
    print('\n'.join([word['base'] for line in lines for word in line if word['pos'] == '動詞']))