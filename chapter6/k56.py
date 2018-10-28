import xml.etree.ElementTree as ET
from pprint import pprint

class Coreference:
    def __init__(self,representative_mention):
        self.representative = representative_mention
        self.coreferences = []

    def __str__(self):
        return '{}\t{}'.format(self.representative, ','.join(['({}:{}:{})'.format(sent_id, start, end) for (sent_id, start, end) in self.coreferences]))

def xml2sentences():
    root = ET.parse('../../../data/nlp.txt.xml')
    sentences = [[word.findtext('word') for word in sentence.iterfind('./tokens/token')] for sentence in root.iterfind('./document/sentences/sentence')]
    return sentences



if __name__ == '__main__':
    root = ET.parse('../../../data/nlp.txt.xml')
    Coreferences = []
    for coreference in root.iterfind('./document/coreference/coreference'):
        for mention in coreference:
            if 'representative' in mention.attrib.keys():
                tmp = Coreference(mention.findtext('text'))
            else:
                tmp.coreferences.append((int(mention.findtext('sentence')) -1, int(mention.findtext('start')) -1, int(mention.findtext('end')) -1))
        Coreferences.append(tmp)

    sentences = xml2sentences()

    for refer in Coreferences:
        for (sent_id, start, end) in refer.coreferences:
            mention = sentences[sent_id][start:end]

            sentences[sent_id] = sentences[sent_id][0:start - 1] + [str(refer.representative + '({})'.format(' '.join(mention)))] + sentences[sent_id][end + 1:]
    [print('\n'.join([' '.join(sequence) for sequence in sentences]))]