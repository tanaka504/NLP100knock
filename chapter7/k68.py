from k64 import access_DB
import pymongo, json


def sort_with_rating(tag='dance'):
    db, collection = access_DB()

    print('Ranking with {} rating'.format(tag))

    result = collection.find({'tags.value':tag})
    result.sort('rating.count', pymongo.DESCENDING)

    [print('{}(ID:{})\t{}'.format(info['name'], info['id'], info['rating']['count'])
           if 'rating' in info else '{}(ID:{})\t{}'.format(info['name'], info['id'], '(None)')) for idx, info in enumerate(result[0:10], 1)]


if __name__ == '__main__':
    print('Ranking with tag rating you specify (Default is "dance")')
    print('When you want to end a search, type "exit".')
    while 1:
        tag = input('Input tag : ')
        if tag == 'exit':
            print('bye')
            break
        elif tag == '':
            tag = 'dance'
        sort_with_rating(tag)