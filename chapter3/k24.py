# coding :utf-8

import re
from k20 import extract_json

if __name__ == '__main__':
    file_pattern = re.compile(r'(ファイル|File):(.*?)\|')

    uk_data = extract_json().split('\n')
    print('\n'.join([file_pattern.search(line).group(2) for line in uk_data if file_pattern.match(line)]))