from k64 import access_DB

'''
$mongo
use mongo_artist
db.artist.find({area:'Japan'}).count()
'''

if __name__ == '__main__':
    db, collection = access_DB()
    print(collection.find({'area':'Japan'}).count())