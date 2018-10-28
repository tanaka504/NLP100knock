if __name__ == '__main__':
    with open('../../../data/hightemp.txt','r') as f:
        print(f.read().replace('\t',' '))
