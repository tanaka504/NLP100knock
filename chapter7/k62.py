import plyvel

if __name__ == '__main__':
    db = plyvel.DB('../../../data/artist.ldb')

    [print(k.decode('utf-8')) for k, v in db if v.decode('utf-8') == 'Japan']