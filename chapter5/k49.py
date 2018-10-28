from k41 import get_chunk

if __name__ == '__main__':
    sentences = get_chunk()
    sentence = sentences[5]
    idx_noun = [j for j in range(len(sentence)) if len(sentence[j]._get_morphs('名詞')) > 0]

    for k,idx_x in enumerate(idx_noun[:-1]):
        for idx_y in idx_noun[k+1:]:
            x_route = []
            y_route = []
            x_dst = idx_x
            y_dst = idx_y
            while x_dst != -1:
                x_route.append(x_dst)
                x_dst = sentence[x_dst].dst

            while y_dst != -1:
                y_route.append(y_dst)
                y_dst = sentence[y_dst].dst
            #print('x:',idx_x,'y:',idx_y)
            #print('x:',x_route,'y:',y_route)

            #XとYが衝突するインデックスを調べる
            xy_dump = min(set(x_route) & set(y_route))

            #｜いらない
            if xy_dump == y_route[0]:
                print(sentence[x_route[0]]._noun2xy('X'),end='')
                for x in x_route[1:]:
                    if x == xy_dump:
                        break
                    print(' -> ' + sentence[x]._normalized_surface(),end='')
                print(' -> ' + sentence[xy_dump]._noun2xy('Y'))

            #｜いる
            else:
                print(sentence[x_route[0]]._noun2xy('X'),end='')
                for x in x_route[1:]:
                    if x == xy_dump:
                        break
                    print(' -> ' + sentence[x]._normalized_surface(),end='')

                print(' | ',end='')

                print(sentence[y_route[0]]._noun2xy('Y'),end='')
                for y in y_route[1:]:
                    if y == xy_dump:
                        break
                    print(' -> ' + sentence[y]._normalized_surface(),end='')

                print(' | ',end='')
                print(sentence[xy_dump]._normalized_surface())
