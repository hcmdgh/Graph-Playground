from .imports import * 
from .graph import * 
from torch_geometric.datasets import Planetoid, CitationFull
import torch_geometric.transforms as T

__all__ = ['load_pyg_dataset']


def convert_pyg_homo_graph(g: pygdata.Data,
                           add_self_loop: bool = False) -> HomoGraph:
    node_attr_dict = { 
        'feat': g.x, 
        'label': g.y,
    }
    
    if 'train_mask' in g:
        node_attr_dict['train_mask'] = g.train_mask
        node_attr_dict['val_mask'] = g.val_mask
        node_attr_dict['test_mask'] = g.test_mask
    
    homo_graph = HomoGraph(
        num_nodes = g.num_nodes,
        edge_index = tuple(g.edge_index),
        node_attr_dict = node_attr_dict, 
        num_classes = int(torch.max(g.y)) + 1, 
    )
    
    if add_self_loop:
        homo_graph.add_self_loop()
        
    return homo_graph


def load_pyg_dataset(name: str,
                     root: str = '/home/Dataset/PyG',
                     normalize: bool = False,
                     add_self_loop: bool = False) -> Union[HomoGraph, HeteroGraph]:
    if name == 'Cora':
        g = Planetoid(
            root = root,
            name = 'Cora',
            transform = T.NormalizeFeatures() if normalize else None,
        )[0]
        
        return convert_pyg_homo_graph(g, add_self_loop=add_self_loop)
    elif name == 'CiteSeer':
        g = Planetoid(
            root = root,
            name = 'CiteSeer',
            transform = T.NormalizeFeatures() if normalize else None,
        )[0]
        
        return convert_pyg_homo_graph(g, add_self_loop=add_self_loop)
    elif name == 'PubMed':
        g = Planetoid(
            root = root,
            name = 'PubMed',
            transform = T.NormalizeFeatures() if normalize else None,
        )[0]

        return convert_pyg_homo_graph(g, add_self_loop=add_self_loop)
    elif name == 'DBLP':
        g = CitationFull(
            root = root,
            name = 'dblp',
            transform = T.NormalizeFeatures() if normalize else None,
        )[0]

        return convert_pyg_homo_graph(g, add_self_loop=add_self_loop)
    else:
        raise AssertionError 
    