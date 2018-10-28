# coding: utf-8

import json

def extract_json():
    with open('../../../data/jawiki-country.json','r') as f:
        data = f.read().split('\n')
        data.remove('')
        for line in data:
            json_data = json.loads(line)
            if json_data['title'] == 'イギリス': return json_data['text']


if __name__ == '__main__':
    uk_data = extract_json()
    print(uk_data)