from util import * 
from dgl.data.citation_graph import CoraGraphDataset

__all__ = [
    'load_Cora_dataset',
]


def load_Cora_dataset() -> dgl.DGLGraph:
    dataset = CoraGraphDataset()
    graph = dataset[0]
    
    return graph 
