import dgl 
import pickle 
import torch 
from ogb.nodeproppred import DglNodePropPredDataset

YEAR_RANGE = (2019, 2019)


def main():
    dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='/home/Dataset/DGL')

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    g, label = dataset[0]
    label = label.squeeze() 

    num_nodes = g.num_nodes() 
    year_th = g.ndata['year'] = g.ndata['year'].squeeze()
    g.ndata['label'] = label 
    
    print(g)
    print(year_th.min(), year_th.max())
    
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[(YEAR_RANGE[0] <= year_th) & (year_th <= YEAR_RANGE[1])] = True
    
    g = dgl.node_subgraph(g, nodes=mask)
    print(g)
    print(g.ndata['label'].unique())
    
    mask1 = torch.zeros(g.num_nodes(), dtype=torch.bool)
    mask1[g.ndata['year'] == YEAR_RANGE[0]] = True 
    mask2 = ~mask1 
    
    g.ndata['train_mask'] = mask1 
    g.ndata['val_mask'] = mask2

    print(f"train cnt: {mask1.sum()}, val cnt: {mask2.sum()}")
    
    dgl.save_graphs(f'/home/gh/Dataset/DGL/ogbn_arxiv/g_{YEAR_RANGE[0]}_{YEAR_RANGE[1]}.dgl', [g])

if __name__ == '__main__':
    main() 
