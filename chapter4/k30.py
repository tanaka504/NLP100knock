import pickle


def read_sentence():
    with open('../../../data/neko.txt.mecab', 'r') as f:
        tmp = []
        sentences = []
        data = f.read().split('\n')
        for line in data:
            if line == 'EOS':
                if len(tmp) > 0:
                    sentences.append(tmp)
                tmp = []
            else:
                col1 = line.split('\t')
                assert len(col1) > 1, line
                col2 = col1[1].split(',')

                pos_dic = {'surface':col1[0],
                           'base':col2[6],
                           'pos':col2[0],
                           'pos2':col2[1]}

                tmp.append(pos_dic)
        return sentences

if __name__ == '__main__':
    lines = read_sentence()
    with open('../../../data/neko.list', 'wb') as f:
        pickle.dump(lines, f)