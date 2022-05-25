from util import * 
from ogb.nodeproppred import DglNodePropPredDataset

DATASET_ROOT = '/home/gh/Dataset/ogbn-arxiv'


def load_ogbn_arxiv_dataset():

    dataset = DglNodePropPredDataset('ogbn-arxiv', root=DATASET_ROOT)

    graph, node_labels = dataset[0]
    # Add reverse edges since ogbn-arxiv is unidirectional.
    graph = dgl.add_reverse_edges(graph)
    graph.ndata['label'] = node_labels[:, 0]

    node_features = graph.ndata['feat']
    num_features = node_features.shape[1]
    num_classes = (node_labels.max() + 1).item()

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']  