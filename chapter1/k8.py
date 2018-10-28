import re

def cipher(word):
    en_lower = re.compile(r'^[a-z]+$')
    return ''.join([chr(219 - ord(c)) if en_lower.match(c) else c for c in word])

if __name__ == '__main__':
    print(cipher('AaAa'))