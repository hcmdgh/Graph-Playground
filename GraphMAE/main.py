from .pipeline import * 
from util import * 

if __name__ == '__main__':
    pipeline = GraphMAE_pipeline(
        graph = load_dgl_dataset('cora'),
        seed = 142,
        raw_feat_classification = True, 
    )

    pipeline.run() 
