import xml.etree.ElementTree as ET

if __name__ == '__main__':
    root = ET.parse('../../../data/nlp.txt.xml')
    [print(token.findtext('word')) for token in root.iterfind('./document/sentences/sentence/tokens/token[NER="PERSON"]')]