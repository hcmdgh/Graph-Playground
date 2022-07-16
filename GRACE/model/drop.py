from dl import * 

__all__ = ['drop_feature', 'drop_edge']


def drop_feature(feat: FloatTensor,
                 drop_prob: float = 0.3) -> FloatTensor:
    feat = feat.clone() 
    assert feat.ndim == 2 
    
    feat_dim = feat.shape[-1]
    drop_cnt = int(feat_dim * drop_prob)
    drop_idx = torch.randperm(feat_dim)[:drop_cnt]
    
    feat[:, drop_idx] = 0.
    
    return feat 


def drop_edge(g: dgl.DGLGraph,
              drop_prob: float) -> dgl.DGLGraph:
    g = g.clone()
    
    g = dgl.DropEdge(drop_prob)(g)
    
    return g 
