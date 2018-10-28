import re

split_pattern = re.compile(r'([.;:?!])\s([A-Z])')
def get_sentences():
    with open('../../../data/nlp.txt','r') as f:
        data = list(filter(lambda x:len(x)>0, split_pattern.sub(r'\1\n\2',f.read()).split('\n')))
        return data



if __name__ == '__main__':
    sentences = get_sentences()
    print(sentences)