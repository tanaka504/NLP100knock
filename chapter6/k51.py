from k50 import get_sentences

def get_words():
    sentences = get_sentences()
    sentences = list(map(lambda x: x.replace(' ', '\n'), sentences))
    return '\n\n'.join([sentence for sentence in sentences])

if __name__ == '__main__':
    sentences = get_words()
    print(sentences)