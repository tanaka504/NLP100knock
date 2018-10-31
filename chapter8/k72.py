from nltk import stem
from k71 import create_stopword

def extract_feature():
    stopwd = create_stopword()
    with open('../../../data/polaritydata/sentiment.txt', 'r') as raw_f, \
        open('../../../data/polaritydata/stemmed.txt', 'w') as out_f:
        data = raw_f.read().split('\n')
        stemmed = [[line.split(' ')[0]] + [stemming(word) for word in line.split(' ')[1:] if not word in stopwd] for line in data]
        out_f.write('\n'.join([' '.join(line) for line in stemmed]))

def stemming(word):
    stemmer = stem.PorterStemmer()
    try:
        return stemmer.stem(word)
    except:
        return word

if __name__ == '__main__':
    extract_feature()