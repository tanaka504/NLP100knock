import pickle

if __name__ == '__main__':
    with open('../../../data/neko.list', 'rb') as f:
        lines = pickle.load(f)
    result = []
    for line in lines:
        nouns = []
        for word in line:
            if word['pos'] == '名詞':
                nouns.append(word['surface'])
            else:
                if len(nouns) > 1:
                    result.append(nouns)
                nouns = []
        if len(nouns) > 1:
            result.append(nouns)
    print('\n'.join([''.join([w for w in words]) for words in result]))