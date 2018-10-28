from chapter1.k5 import char_n_gram

if __name__ == '__main__':
    text1 = 'paraparaparadise'
    text2 = 'paragraph'
    subset_word = {'se'}

    X = set(char_n_gram(text1, 2))
    Y = set(char_n_gram(text2, 2))
    print(X | Y)
    print(X & Y)
    print(X - Y)
    print('"se" is in X:{} Y :{}'.format(str(subset_word <= X), str(subset_word <= Y)))