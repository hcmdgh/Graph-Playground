from .pipeline import * 
from util import * 

if __name__ == '__main__':
    GCN_pipeline(
        graph = load_dgl_dataset('pubmed'), 
    )
