from dl import * 


def drop_edge(g: dgl.DGLGraph, 
              p: float) -> dgl.DGLGraph:
    edge_index = g.edges() 
    num_nodes = g.num_nodes() 
    num_edges = len(edge_index[0])
    
    edge_mask = torch.bernoulli(torch.full(fill_value=1-p, size=[num_edges])).to(torch.bool)

    dropped_edge_index = (edge_index[0][edge_mask], edge_index[1][edge_mask])
    
    dropped_g = dgl.graph(dropped_edge_index, num_nodes=num_nodes)
    
    dropped_g = dgl.remove_self_loop(dropped_g)
    dropped_g = dgl.add_self_loop(dropped_g)

    return dropped_g 


def drop_edge_weighted(g: dgl.DGLGraph, 
                       edge_weight: FloatTensor, 
                       p: float, 
                       threshold: float = 1.) -> dgl.DGLGraph:
    edge_index = g.edges() 
    num_nodes = g.num_nodes() 
    num_edges = len(edge_index[0])
    
    assert edge_weight.shape == edge_index[0].shape 
    
    edge_weight = edge_weight / edge_weight.mean() * p 
    edge_weight[edge_weight > threshold] = threshold

    edge_mask = torch.bernoulli(1 - edge_weight).to(torch.bool)

    dropped_edge_index = (edge_index[0][edge_mask], edge_index[1][edge_mask])
    
    dropped_g = dgl.graph(dropped_edge_index, num_nodes=num_nodes)
    
    dropped_g = dgl.remove_self_loop(dropped_g)
    dropped_g = dgl.add_self_loop(dropped_g)

    return dropped_g 


def calc_feat_drop_weight(g: dgl.DGLGraph,
                          feat: FloatTensor) -> FloatTensor:
    deg = g.in_degrees().to(torch.float32)
                          
    feat = feat.to(torch.bool).to(torch.float32)

    weight = torch.log(feat.T @ deg)
    
    out = (weight.max() - weight) / (weight.max() - weight.mean())

    return out 


def calc_edge_drop_weight(g: dgl.DGLGraph) -> FloatTensor:
    edge_index = g.edges() 
    num_nodes = g.num_nodes() 
    num_edges = len(edge_index[0])
    
    deg = g.in_degrees().to(torch.float32)
    assert deg.shape == (num_nodes,)
    
    dest_deg = deg[edge_index[1]]
    assert torch.all(dest_deg > 0)
    
    log_dest_deg = torch.log(dest_deg)

    # TODO mean or min?
    edge_weight = (log_dest_deg.max() - log_dest_deg) / (log_dest_deg.max() - log_dest_deg.mean())
    assert edge_weight.shape == (num_edges,)

    return edge_weight


def drop_feat(feat: FloatTensor, 
              p: float) -> FloatTensor:
    feat_dim = feat.shape[-1]
    
    drop_mask = torch.bernoulli(torch.full(fill_value=p, size=[feat_dim])).to(torch.bool)
    
    feat = feat.clone()
    feat[:, drop_mask] = 0. 

    return feat


def drop_feat_weighted(feat: FloatTensor,
                       feat_weight: FloatTensor,  
                       p: float,
                       threshold: float = 0.7) -> FloatTensor:
    feat_dim = feat.shape[-1]
    assert feat_weight.shape == (feat_dim,)
    
    feat_weight = feat_weight / feat_weight.mean() * p 
    feat_weight[feat_weight > threshold] = threshold 
    
    drop_mask = torch.bernoulli(feat_weight).to(torch.bool)
    
    feat = feat.clone()
    feat[:, drop_mask] = 0. 

    return feat
