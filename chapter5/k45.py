from k41 import get_chunk

if __name__ == '__main__':
    sentences = get_chunk()
    #for sentence in sentences:
    sentence = sentences[5]
    for chunk in sentence:
        verbs = chunk._get_morphs('動詞')
        if len(verbs) < 1: continue

        josi = [sentence[src]._get_kaku() for src in chunk.srcs]
        if len(josi) < 1: continue

        print('{}\t{}'.format(verbs[0].base, ' '.join(sorted(josi))))