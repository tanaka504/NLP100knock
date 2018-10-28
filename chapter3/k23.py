# coding :utf-8

import re
from k20 import extract_json


if __name__ == '__main__':
    section_pattern = re.compile(r'^(=+)(.*?)(=+)$')

    uk_data = extract_json().split('\n')
    print('\n'.join(['{}:{}'.format(section_pattern.search(line).group(2),\
        len(section_pattern.search(line).group(1))) for line in uk_data if section_pattern.match(line)]))
