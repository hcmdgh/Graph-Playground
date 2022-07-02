from .pipeline import * 
from util import * 

if __name__ == '__main__':
    GAT_pipeline(
        graph = load_dgl_dataset('cora'),
        GAT_param = GAT.Param(
            in_dim = -1,
            out_dim = -1, 
        ),  
    )
