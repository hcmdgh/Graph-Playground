from .pipeline import * 
from util import * 

if __name__ == '__main__':
    DGI_pipeline(
        graph = load_dgl_dataset('cora'),
    )