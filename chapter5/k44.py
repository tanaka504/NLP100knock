from k41 import get_chunk
from graphviz import Digraph

def Edge2Digraph(edge_list):
    G = Digraph(format='png')
    G.attr('node',shape='circle')

    for edge in edge_list:
        G.node(str(edge[0]),str(edge[1]))
    for edge in edge_list:
        if not edge[2] == -1:
            G.edge(str(edge[0]),str(edge[2]))
    print(G)
    G.render('../result/result')

if __name__ == '__main__':
    sentences = get_chunk()
    Edge2Digraph([(idx, chunk._normalized_surface(), chunk.dst) for idx, chunk in enumerate(sentences[5])])