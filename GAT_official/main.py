from .pipeline import * 
from util import * 

if __name__ == '__main__':
    g = dgl.load_graphs('./DBLP/output/dblp_part.dgl')[0][0]
    
    year2cnt: dict[int, int] = defaultdict(int)
    
    for year in g.ndata['year']:
        year = int(year)
        year2cnt[year] += 1 
        
    print(year2cnt)
    
    g.ndata['train_mask'] = g.ndata['year'] == 2015
    g.ndata['val_mask'] = g.ndata['year'] == 2016
    g.ndata['test_mask'] = g.ndata['year'] >= 2017
    
    GAT_pipeline(
        graph = g,
        GAT_param = GAT.Param(
            in_dim = -1,
            out_dim = -1, 
            feat_dropout = 0.6,
            attn_dropout = 0.6,
        ),  
    )
