import xml.etree.ElementTree as ET
import pydot_ng as pydot

def edges2graph(edge_list):
    graph = pydot.Dot(graph_type='digraph')

    for edge in edge_list:
        id1 = str(edge[0][0])
        label1 = str(edge[0][1])
        id2 = str(edge[1][0])
        label2 = str(edge[1][1])

        graph.add_node(pydot.Node(id1,label = label1))
        graph.add_node(pydot.Node(id2,label = label2))

        graph.add_edge(pydot.Edge(id1,id2))
    return graph

if __name__ == '__main__':
    root = ET.parse('../../../data/nlp.txt.xml')

    for sentence in root.iterfind('./document/sentences/sentence'):
        sent_id = int(sentence.get('id'))
        edges = []

        for dep in sentence.iterfind('./dependencies[@type="collapsed-dependencies"]/dep'):
            if dep.get('type') != 'punct':
                govr = dep.find('./governor')
                dept = dep.find('./dependent')
                edges.append(((govr.get('idx'),govr.text),(dept.get('idx'),dept.text)))

        if len(edges) > 0:
            graph = edges2graph(edges)
            graph.write_png('../result/{}.png'.format(sent_id))