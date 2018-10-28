# coding :utf-8

import re
from k20 import extract_json

if __name__ == '__main__':
    category_pattern = re.compile(r'^\[\[Category:(.*?)(|\|.*)\]\]')

    uk_data = extract_json().split('\n')
    print('\n'.join([category_pattern.search(line).group(1) for line in uk_data \
               if category_pattern.match(line)]))
