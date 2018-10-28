import json, pymongo
from pymongo import MongoClient
from io import StringIO

def create_artistDB():
    client = MongoClient()
    db = client.mongo_artist
    collection = db.artist

    with open('../../../data/artist.json', 'r') as f:
        data = f.read().split('\n')
        data.remove('')

        collection.insert_many([json.loads(line) for line in data])
        collection.create_index([('name', pymongo.ASCENDING)])
        collection.create_index([('aliases.name', pymongo.ASCENDING)])
        collection.create_index([('tags.value', pymongo.ASCENDING)])
        collection.create_index([('rating.value', pymongo.ASCENDING)])

def access_DB():
    client = pymongo.MongoClient()
    db = client.mongo_artist
    collection = db.artist

    print('access to DB:', db.name)
    print('access to collection:', collection.name)

    return db, collection



if __name__ == '__main__':
    create_artistDB()