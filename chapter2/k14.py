import sys

if __name__ == '__main__':
    args = sys.argv

    assert len(args) > 1, 'How long lines do you want?'
    assert args[1].isdigit(), 'Input must be integer'
    with open('../../../data/hightemp.txt', 'r') as f:
        data = f.read().split('\n')
        print('\n'.join(data[:int(args[1])]))
