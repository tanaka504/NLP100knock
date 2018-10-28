from k41 import get_chunk

if __name__ == '__main__':
    sentences = get_chunk()
    [print('{}\t{}'.format(chunk._normalized_surface(), sentence[chunk.dst]._normalized_surface()))
     for sentence in sentences for chunk in sentence if not chunk.dst == -1 and chunk._chk_pos('名詞') and sentence[chunk.dst]._chk_pos('動詞')]