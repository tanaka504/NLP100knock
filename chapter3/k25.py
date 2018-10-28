# coding :utf-8

import re
from k20 import extract_json
from pprint import pprint


info_pattern = re.compile(r'^\|(.*?)\s=\s(.*)')
def extract_basic_info():
    uk_data = extract_json().split('\n')
    basic_info = {info_pattern.search(line).group(1): info_pattern.search(line).group(2) \
                  for line in uk_data if info_pattern.match(line)}
    return basic_info

if __name__ == '__main__':
    pprint(extract_basic_info())