if __name__ == '__main__':
    f = open('../../../data/hightemp.txt','r')
    f1 = open('../../../data/col1.txt','w')
    f2 = open('../../../data/col2.txt', 'w')

    data = f.read().split('\n')
    data.remove('')
    for line in data:
        cols = line.split('\t')
        f1.write(cols[0] + '\n')
        f2.write(cols[1] + '\n')

    f.close()
    f1.close()
    f2.close()