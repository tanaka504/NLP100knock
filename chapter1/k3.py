import re

if __name__ == '__main__':
    text = 'Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.'
    tokens = re.compile(r'[,|\.]')
    text = tokens.sub('',text)

    words = text.split(' ')
    print('{}.{}'.format(str(len(words[0])), ''.join(str(len(c)) for c in words[1:])))