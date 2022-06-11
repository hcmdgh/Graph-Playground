from util import * 
from .pipeline import * 
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset


def preprocess_dataset():
    dataset = PubmedGraphDataset()
    g = dataset[0]
    
    homo_graph = HomoGraph(
        num_nodes = g.num_nodes(),
        edge_index = g.edges(),
        node_attr_dict = {
            'feat': g.ndata['feat'],
            'label': g.ndata['label'],
            'train_mask': g.ndata['train_mask'],
            'val_mask': g.ndata['val_mask'],
            'test_mask': g.ndata['test_mask'],
        },
        num_classes = 7, 
    )
    
    homo_graph.save_to_file('/home/Dataset/DGL/Pubmed.pt')

    
def main():
    GCN_pipeline(
        homo_graph_path = '/home/Dataset/DGL/Cora.pt',
    )    


if __name__ == '__main__':
    main() 
