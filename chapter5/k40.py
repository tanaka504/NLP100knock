

class Morph:
    def __init__(self, surface, base, pos, pos2):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos2 = pos2

    def __str__(self):
        return 'surface:{}\tbase:{}\tpos:{}\tpos2:{}'\
            .format(self.surface, self.base, self.pos, self.pos2)

def read_sentence():
    with open('../../../data/neko.txt.cabocha', 'r') as f:
        tmp = []
        sentences = []
        data = f.read().split('\n')
        data.remove('')
        for line in data:
            if line == 'EOS':
                if len(tmp) > 0:
                    sentences.append(tmp)
                tmp = []
            else:
                if line[0] == '*':
                    continue
                col1 = line.split('\t')
                assert len(col1) > 1, line
                col2 = col1[1].split(',')

                tmp.append(Morph(col1[0], col2[6], col2[0], col2[1]))
        return sentences

if __name__ == '__main__':
    sentece = read_sentence()
    [print(morph) for morph in sentece[3]]