import sys
from pprint import pprint

if __name__ == '__main__':
    args = sys.argv
    with open('../../../data/hightemp.txt', 'r') as f:
        data = f.read().split('\n')
        N = len(data) // int(args[1])
        data = [data[i:i+N] for i in range(0,len(data),N)]
        pprint(data)