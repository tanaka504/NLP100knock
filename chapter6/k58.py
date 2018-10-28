import xml.etree.ElementTree as ET
from pprint import pprint

if __name__ == '__main__':
    root = ET.parse('../../../data/nlp.txt.xml')
    ans_list = []
    nsubj_list = []
    dobj_list = []
    for sentence in root.iterfind('./document/sentences/sentence'):
        for dep in sentence.iterfind('./dependencies[@type="collapsed-dependencies"]/dep'):

            dep_type = dep.get('type')
            if dep_type == 'nsubj' or dep_type == 'dobj':
                govr = dep.find('./governor')
                dept = dep.find('./dependent')
                if dep_type == 'nsubj':
                    nsubj_list.append((govr.get('idx'),govr.text,dept.text))
                else:
                    dobj_list.append((govr.get('idx'),govr.text,dept.text))

            for nsubj in nsubj_list:
                for dobj in dobj_list:
                    if nsubj[0] == dobj[0]:
                        ans_list.append((nsubj[2],nsubj[1],dobj[2]))
                        nsubj_list = []
                        dobj_list = []
    pprint(list(set(ans_list)))