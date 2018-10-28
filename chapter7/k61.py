import plyvel

if __name__ == '__main__':
    db = plyvel.DB('../../../data/artist.ldb')
    print(db.get(u'Queen'.encode('utf-8')))