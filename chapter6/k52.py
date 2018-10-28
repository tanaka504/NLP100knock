from k51 import get_words
from nltk.stem.porter import PorterStemmer

if __name__ == '__main__':
    sentences = get_words().split('\n')
    stemmer = PorterStemmer()
    [print('{}\t{}'.format(word, stemmer.stem(word))) for word in sentences]