def word_n_gram(text, n):
    result = []
    words = text.split(' ')
    max_idx = len(words)
    start = 0
    end = n
    while end <= max_idx:
        result.append(' '.join(words[start:end]))
        start += 1
        end += 1
    return result

def char_n_gram(text, n):
    result = []
    chars = list(text)
    max_idx = len(chars)
    start = 0
    end = n
    while end <= max_idx:
        result.append(''.join(chars[start:end]))
        start += 1
        end += 1
    return result

if __name__ == '__main__':
    text = 'I am an NLPer'
    word_bi_gram = word_n_gram(text, 2)
    char_bi_gram = char_n_gram(text, 2)
    print(word_bi_gram)
    print(char_bi_gram)