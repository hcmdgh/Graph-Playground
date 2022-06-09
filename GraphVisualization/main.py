from .util import * 
from util import * 
from dgl.data import CoraGraphDataset


def main():
    with open('./test.gexf', 'r', encoding='utf-8') as fp:
        text = fp.read() 
        
    text = re.sub(r'tensor\((.+)\)', r'\1', text)

    print(text[:1000])
    
    with open('./test.gexf', 'w', encoding='utf-8') as fp:
        fp.write(text)
    
    exit() 
    
    dataset = CoraGraphDataset()
    graph = dataset[0]

    N = graph.num_nodes() 
    label = graph.ndata['label']
    
    G = graph.to_networkx(node_attrs=['label'])
    
    # visualize(label, G)
    
    nx.write_gexf(G, './test.gexf')


if __name__ == '__main__':
    main() 
