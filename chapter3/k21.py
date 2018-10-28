# coding :utf-8

from k20 import extract_json

if __name__ == '__main__':
    uk_data = extract_json().split('\n')
    print('\n'.join([line for line in uk_data if 'Category' in line]))


