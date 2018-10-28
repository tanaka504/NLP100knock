import xml.etree.ElementTree as ET
import re

pattern = re.compile(r"^\(  (.*?) \s (.*) \)$",re.VERBOSE + re.DOTALL)

def parse2NP(str,np_list):
    match = pattern.match(str)
    tag = match.group(1)
    value = match.group(2)

    depth = 0
    chunk = ''
    words = []
    for c in value:
        if c == '(':
            chunk += c
            depth += 1
        elif c == ')':
            chunk += c
            depth -= 1
            if depth == 0:
                words.append(parse2NP(chunk,np_list))
                chunk = ''
        else:
            if not (depth == 0 and c == ' '):
                chunk += c
    if chunk != '':
        words.append(chunk)

    result = ' '.join(words)
    if tag == 'NP':
        np_list.append(result)

    return result


if __name__ == '__main__':
    root = ET.parse('../../../data/nlp.txt.xml')

    for parse in root.iterfind('./document/sentences/sentence/parse'):
        result = []
        parse2NP(parse.text.strip(),result)
        print(result)