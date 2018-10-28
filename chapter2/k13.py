if __name__ == '__main__':
    f1 = open('../../../data/col1.txt', 'r')
    f2 = open('../../../data/col2.txt', 'r')
    f = open('../../../data/merged.txt','w')

    data1 = f1.read().split('\n')
    data2 = f2.read().split('\n')
    data1.remove('')
    data2.remove('')

    for line1, line2 in zip(data1, data2):
        f.write('{}\t{}\n'.format(line1, line2))

    f1.close()
    f2.close()
    f.close()