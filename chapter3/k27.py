# coding :utf-8

import re
from k20 import extract_json
from k25 import *
from k26 import *

link_pattern = re.compile(r'(\[){2,5}(.*?)(\]){2,5}')
def remove_link_markup():
    uk_data = extract_json().split('\n')
    basic_info = {info_pattern.search(line).group(1): \
                    link_pattern.sub(r'\2',emp_pattern.sub('',info_pattern.search(line).group(2))) \
                  for line in uk_data if info_pattern.match(line)}
    return basic_info


if __name__ == '__main__':
    print(extract_basic_info())
    print(remove_emp_markup())
    print(remove_link_markup())