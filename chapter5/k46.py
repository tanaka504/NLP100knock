from k41 import get_chunk

if __name__ == '__main__':
    sentences = get_chunk()
    #for sentence in sentences:
    sentence = sentences[5]
    for chunk in sentence:
        verbs = chunk._get_morphs('動詞')
        if len(verbs) < 1: continue

        chunks_inc_josi = []
        for src in chunk.srcs:
            if len(sentence[src]._get_kaku()) > 0:
                chunks_inc_josi.append(sentence[src])
        if len(chunks_inc_josi) < 1: continue

        print('{}\t{}\t{}'.format(
            verbs[0].base,
            ' '.join([chunk._get_kaku() for chunk in chunks_inc_josi]),
            ' '.join([chunk._normalized_surface() for chunk in chunks_inc_josi])
        ))
