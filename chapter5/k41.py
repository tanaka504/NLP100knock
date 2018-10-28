from k40 import Morph
import re
import pickle

class Chunk:
    def __init__(self):
        self.morphs = []
        self.dst = -1
        self.srcs = []

    def __str__(self):
        sentence = ''.join([word.surface for word in self.morphs])
        return '{}\tdst:{}\tsrcs:{}'.format(sentence, self.dst, self.srcs)

    def _normalized_surface(self):
        return ''.join([word.surface for word in self.morphs])

    def _chk_pos(self, pos):
        for word in self.morphs:
            if word.pos == pos:
                return True
        return False

    def _get_morphs(self,pos,pos2=None):
        if pos2 is None:
            return [morph for morph in self.morphs if morph.pos == pos]
        else:
            return [morph for morph in self.morphs if morph.pos == pos and morph.pos2 == pos2]

    def _get_kaku(self):
        josi = self._get_morphs('助詞')
        if len(josi) > 1:
            kaku = self._get_morphs('助詞', '格助詞')
            if len(kaku) > 0:
                josi = kaku
        if len(josi) > 0:
            return josi[-1].surface
        else:
            return ''

    def _get_sahen_wo(self):
        for i, morph in enumerate(self.morphs[0:-1]):
            if (morph.pos == '名詞') and (morph.pos2 == 'サ変接続') and (self.morphs[i + 1].pos == '助詞') and (
                    self.morphs[i + 1].surface == 'を'):
                return morph.surface + self.morphs[i + 1].surface

        return ''

    def _noun2xy(self, xy, dst=False):
        result = ''
        for morph in self.morphs:
            if morph.pos != '記号':
                if morph.pos == '名詞':
                    result += xy
                    if dst:
                        return result
                    xy = ''
                else:
                    result += morph.surface

        return result

def get_chunk():
    chunk = {}
    sentences = []
    with open('../../../data/neko.txt.cabocha', 'r') as f:
        data = f.read().split('\n')
        data.remove('')
        for line in data:
            if line[0] == '*':
                chunk_col = line.split(' ')
                dst = int(re.search(r'(.*?)D',chunk_col[2]).group(1))
                idx = int(chunk_col[1])
                if idx not in chunk: chunk[idx] = Chunk()
                chunk[idx].dst = dst

                if not dst == -1:
                    if dst not in chunk: chunk[dst] = Chunk()
                    chunk[dst].srcs.append(idx)

            elif line == 'EOS':
                if len(chunk) > 0: sentences.append(list(zip(*sorted(chunk.items(), key=lambda x:x[0])))[1])
                chunk.clear()

            else:
                col1 = line.split('\t')
                col2 = col1[1].split(',')
                chunk[idx].morphs.append(Morph(col1[0],
                                               col2[6],
                                               col2[0],
                                               col2[1]))
        return sentences

if __name__ == '__main__':
    sentence = get_chunk()
    [print(chunk) for chunk in sentence[6]]