from pipeline import * 
from dl import * 

if __name__ == '__main__':
    GraphMAE_pipeline(
        graph = load_dgl_dataset('cora'),
    )
