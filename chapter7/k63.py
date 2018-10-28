import plyvel
import json


def create_DB():
    db = plyvel.DB('../../../data/artist_tag.ldb', create_if_missing=True)
    with open('../../../data/artist.json', 'r') as f:
        data = f.read().split('\n')
        data.remove('')
        for line in data:
            artist_data = json.loads(line)
            if 'tags' in artist_data.keys() and 'name' in artist_data.keys():
                db.put(artist_data['name'].encode('utf-8'), json.dumps(artist_data['tags']).encode('utf-8'))
    return db

if __name__ == '__main__':
    db = create_DB()
    [print(k.decode('utf-8'), v.decode('utf-8')) for k, v in db]