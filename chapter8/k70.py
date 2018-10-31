import random

def concat_data():
    with open('../../../data/polaritydata/rt-polarity.neg', 'r') as neg_f, \
        open('../../../data/polaritydata/rt-polarity.pos', 'r') as pos_f, \
        open('../../../data/polaritydata/sentiment.txt', 'w') as out_f:
        neg_data = neg_f.read().split('\n')
        pos_data = pos_f.read().split('\n')
        neg_data = ['-1 ' + line for line in neg_data if not line == '']
        pos_data = ['+1 ' + line for line in pos_data if not line == '']
        out_data = neg_data + pos_data
        random.shuffle(out_data)

        out_f.write('\n'.join(out_data))

if __name__ == '__main__':
    concat_data()