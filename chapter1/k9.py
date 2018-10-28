import random

def shuffle_word(word):
    chars = list(word)
    shuffle_list = chars[1:-1]
    random.shuffle(shuffle_list)
    return ''.join(chars[0:1] + shuffle_list + chars[-1:])

if __name__ == '__main__':
    text = "I couldn't believe that I could actually understand what I was reading : " \
           "the phenomenal power of the human mind ."

    words = text.split(' ')
    print(' '.join([word if len(word) <= 4 else shuffle_word(word) for word in words]))
