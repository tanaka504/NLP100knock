if __name__ == '__main__':
    text = 'Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations ' \
           'Might Also Sign Peace Security Clause. Arthur King Can.'

    word1_idx = [1,5,6,7,8,9,15,16,19]

    words = text.split(' ')
    element_dict = { idx : word[0] if idx in word1_idx else word[0:2] for idx, word in enumerate(words, 1)}
    print(element_dict)