from k64 import access_DB
import json
from bson.objectid import ObjectId

def support_ObjectId(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(repr(obj) + ' is not JSON serializable')

def search_artist(alias_name):
    db, collection = access_DB()
    [print('{} : {}'.format(idx, json.dumps(info, indent='\t', ensure_ascii=False, sort_keys=True, default=support_ObjectId)))
     for idx, info in enumerate(collection.find({'aliases.name': alias_name}), 1)]


if __name__ == '__main__':
    print('Search artist from alias')
    print('When you want to end a search, type "exit".')
    while 1:
        aliase_name = input('Input artist alias >> ')

        if aliase_name == 'exit':
            print('bye')
            break
        search_artist(aliase_name)



