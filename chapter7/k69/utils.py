import json, pymongo, re
from pprint import pprint

def access_DB():
    client = pymongo.MongoClient()
    db = client.mongo_artist
    collection = db.artist

    print('access to DB:', db.name)
    print('access to collection:', collection.name)

    return db, collection

def create_table(index):
    index = Missing_Value_Processing(index)

    index['tags_list'] = ','.join([tag['value'] if not tag['value'] == 0 else 'None' for tag in index['tags']])
    index['aliase_name'] = ','.join([aliase['name'] if not aliase['name'] == 0 else 'None' for aliase in index['aliases']])

    return index

def Missing_Value_Processing(index):
    simple_factors = ['name', 'area']
    dict_factors = {
        'tags': ('count', 'value'),
        'aliases': ('name','sort_name'),
                    }

    # value が string の要素の欠損値処理
    for factor in simple_factors:
        if not factor in index.keys():
            index[factor] = 'None'

    # value が list の要素の欠損値処理
    for factor, values in dict_factors.items():
        if not factor in index.keys():
            index[factor] = [{value : 0 for value in values}]

    # rating だけ dict だったので別枠で処理
    if not 'rating' in index.keys():
        index['rating'] = {'count': 0, 'value': 0}

    return index

def retrieve(kwds, type):
    retrieve_key = {
        'by_kwd': 'name',
        'by_tag': 'tags.value'
    }
    db, collection = access_DB()

    kwds = re.split(r'(\s|　)', kwds)
    tmp = [create_table(index) for index in collection.find({retrieve_key[type]: kwds[0]})]

    result = sorted(tmp, key=lambda x:(x['rating']['value'], max([tag['count'] for tag in x['tags']])), reverse=True)
    return result

if __name__ == '__main__':
    result = retrieve('Queen')
    pprint(result)