# coding :utf-8

import requests
from k28 import *
from pprint import pprint

URL = 'https://en.wikipedia.org/w/api.php'

# 再帰でjsonを綺麗にしていたのを見つけたので採用
def json_search(json_data):
    ret_dict = {}
    for k, v in json_data.items():
        if isinstance(v, list):
            for e in v:
                ret_dict.update(json_search(e))
        elif isinstance(v, dict):
            ret_dict.update(json_search(v))
        else:
            ret_dict[k] = v
    return ret_dict

def get_info():
    basic_info = remove_Mediawiki_markup()
    payload = {'action': 'query',
               'titles': 'File:{}'.format(basic_info[u'国旗画像']),
               'prop': 'imageinfo',
               'format': 'json',
               'iiprop': 'url'}
    json_data = requests.get(URL, params=payload).json()
    return json_data

if __name__ == '__main__':
    data = get_info()
    print(json_search(data)['url'])