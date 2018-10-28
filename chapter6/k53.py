import xml.etree.ElementTree as ET

#./corenlp.sh -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref -file ./nlp.txt

if __name__ == '__main__':
    root = ET.parse('../../../data/nlp.txt.xml')
    [print(word.text) for word in root.iter('word')]

