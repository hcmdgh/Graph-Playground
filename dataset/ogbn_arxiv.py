from util import * 
from ogb.nodeproppred import DglNodePropPredDataset

DATASET_ROOT = './dataset/bin/ogbn-arxiv'


def load_ogbn_arxiv_dataset() -> HomoGraph:
    dataset = DglNodePropPredDataset('ogbn-arxiv', root=DATASET_ROOT)

    g, node_labels = dataset[0]
    
    # 无向图
    g = dgl.to_bidirected(g, copy_ndata=True)

    g.ndata['label'] = node_labels[:, 0]
    num_classes = len(torch.unique(node_labels))

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']  
    
    num_nodes = g.num_nodes() 
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_nids] = True
    val_mask[valid_nids] = True
    test_mask[test_nids] = True
    
    assert (train_mask | val_mask | test_mask).all()
    assert (~(train_mask & val_mask & test_mask)).all()

    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    
    homo_graph = HomoGraph.from_dgl(g)
    homo_graph.num_classes = num_classes
    
    return homo_graph
    
    
if __name__ == '__main__':
    load_ogbn_arxiv_dataset()
