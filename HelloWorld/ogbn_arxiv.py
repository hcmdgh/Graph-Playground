from util import * 
from ogb.nodeproppred import DglNodePropPredDataset

dataset = DglNodePropPredDataset(name='ogbn-arxiv', root='/home/Dataset/DGL')

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph, label = dataset[0]

train_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
val_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
test_mask = torch.zeros(graph.num_nodes(), dtype=torch.bool)
train_mask[train_idx] = True 
val_mask[valid_idx] = True 
test_mask[test_idx] = True 
assert torch.all(train_mask | val_mask | test_mask)
assert torch.all(~(train_mask & val_mask & test_mask))

homo_graph = HomoGraph(
    num_nodes = graph.num_nodes(),
    edge_index = graph.edges(),
    node_attr_dict = {
        'year': graph.ndata['year'].view(-1),
        'feat': graph.ndata['feat'],
        'label': label.view(-1), 
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
    },
    num_classes = 40, 
)

homo_graph.save_to_file('/home/Dataset/GengHao/HomoGraph/DGL/ogbn-arxiv.pt')
