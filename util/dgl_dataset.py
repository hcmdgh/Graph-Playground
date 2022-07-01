from .imports import * 
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

__all__ = ['load_dgl_dataset']


def load_dgl_dataset(dataset_name: str) -> Union[dgl.DGLGraph, dgl.DGLHeteroGraph]:
    dataset_name = dataset_name.lower() 
    
    if dataset_name == 'cora':
        dataset = CoraGraphDataset()
        graph = dataset[0]
        
        return graph 
    
    elif dataset_name == 'citeseer':
        dataset = CiteseerGraphDataset()
        graph = dataset[0]
        
        return graph 
    
    elif dataset_name == 'pubmed':
        dataset = PubmedGraphDataset()
        graph = dataset[0]

        return graph 
    
    else:
        raise AssertionError 
