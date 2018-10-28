from k64 import access_DB


if __name__ == '__main__':
    db, collection = access_DB()

    [print(index) for index in collection.find({'name': 'Queen'})]