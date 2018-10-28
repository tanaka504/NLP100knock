import xml.etree.ElementTree as ET

if __name__ == '__main__':
    root = ET.parse('../../../data/nlp.txt.xml')
    [print('{}\t{}\t{}'.format(token.findtext('word'), token.findtext('lemma'), token.findtext('POS')))for token in root.iter('token')]