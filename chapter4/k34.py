import pickle

if __name__ == '__main__':
    with open('../../../data/neko.list', 'rb') as f:
        lines = pickle.load(f)

        print(['{}の{}'.format(line[i-1]['surface'],line[i+1]['surface']) for line in lines \
                             for i in range(1,len(line)-1) if line[i]['surface'] == 'の' and line[i]['pos2'] == '連体化'])