from k41 import get_chunk

if __name__ == '__main__':
    sentences = get_chunk()
    for sentence in sentences:
        for chunk in sentence:
            verbs = chunk._get_morphs('動詞')
            if len(verbs) < 1:
                continue
            chunks_inc_josi = [sentence[src] for src in chunk.srcs if len(sentence[src]._get_kaku()) > 0]
            if len(chunks_inc_josi) < 1: continue

            #サ変接続名詞自身も出力されるので削除
            sahen_wo = ''
            for chunk_src in chunks_inc_josi:
                sahen_wo = chunk_src._get_sahen_wo()
                if len(sahen_wo) > 0:
                    chunk_remove = chunk_src
                    break
            if len(sahen_wo) < 1: continue

            chunks_inc_josi.remove(chunk_remove)

            print('{}\t{}\t{}'.format(
                sahen_wo + verbs[0].base,
                ' '.join([chunk._get_kaku() for chunk in chunks_inc_josi]),
                ' '.join([chunk._normalized_surface() for chunk in chunks_inc_josi])
            ))