from collections import Counter

def create_stopword():
    with open('../../../data/polaritydata/sentiment.txt', 'r') as f:
        data = f.read().split('\n')
        wd_count = Counter([word for line in data for word in line.split(' ')[1:]])

    return {k for k, v in wd_count.items() if v > 1000}


def is_stopwd(word):
    stopwd = create_stopword()
    return True if word in stopwd else False

if __name__ == '__main__':
    print(create_stopword())