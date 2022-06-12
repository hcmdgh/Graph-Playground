from .pipeline import * 
from .model import * 
from util import * 

if __name__ == '__main__':
    HeCo_Pipeline.run(
        hetero_graph_path = '/home/Dataset/GengHao/HeteroGraph/HeCo/ACM.pt',
        positive_sample_mask_path = '/home/Dataset/GengHao/HeteroGraph/HeCo/ACM_mask.npy',
        relation_neighbor_size_dict = {
            ('author', 'ap', 'paper'): 7,
            ('subject', 'sp', 'paper'): 1, 
        },
        metapaths = [['pa', 'ap'], ['ps', 'sp']],
    )
