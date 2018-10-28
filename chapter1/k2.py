if __name__ == '__main__':
    word1 = 'パトカー'
    word2 = 'タクシー'
    print(''.join([c1+c2 for (c1, c2) in zip(word1,word2)]))