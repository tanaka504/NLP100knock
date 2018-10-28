if __name__ == '__main__':
    with open('../../../data/hightemp.txt', 'r') as f:
        col_dict = {}
        data = f.read().split('\n')
        data.remove('')
        for line in data:
            cols = line.split('\t')
            col_dict[(cols[0],cols[1])] = float(cols[2])
        sorted_data = sorted(col_dict.items(), key=lambda x: -x[1])
        print('\n'.join(['{}\t{}\t{}'.format(cols[0][0],cols[0][1],cols[1]) for cols in sorted_data]))
