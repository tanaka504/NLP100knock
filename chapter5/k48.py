from k41 import get_chunk

if __name__ == '__main__':
    sentences = get_chunk()
    sentence = sentences[5]

    for chunk in sentence:
        if (len(chunk._get_morphs('名詞')) > 0):
            print(chunk._normalized_surface(), end='')

            dst = chunk.dst
            while dst != -1:
                print(' -> ' + sentence[dst]._normalized_surface(), end='')
                dst = sentence[dst].dst
            print('\n')