from dl import * 


def normalize_feature(feat: FloatTensor) -> FloatTensor:
    row_sum = feat.sum(dim=-1, keepdim=True)

    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.

    feat = feat * r_inv 
    
    return feat 


def sample_neighbor_from_adj_list(
    adj_list: dict[int, list[int]],
    sample_neighbor_cnt: int,
) -> IntArray:
    sampled_adj_list = dict()
    
    num_nodes = 0 
    
    for src_nid, dest_nids in adj_list.items():
        num_nodes = max(num_nodes, src_nid + 1)
        
        if len(dest_nids) < sample_neighbor_cnt:
            sampled_nids = random.choices(dest_nids, k=sample_neighbor_cnt)
        else:
            sampled_nids = random.sample(dest_nids, k=sample_neighbor_cnt)
            
        sampled_adj_list[src_nid] = sampled_nids 

    adj_arr = np.full([num_nodes, sample_neighbor_cnt], -1, dtype=np.int64)
    
    for src_nid in range(num_nodes):
        for j, dest_nid in enumerate(sampled_adj_list[src_nid]):
            adj_arr[src_nid, j] = dest_nid
            
    assert np.min(adj_arr) >= 0 
        
    return adj_arr


def sample_neighbor_from_graph(
    graph: dgl.DGLGraph,
    sample_neighbor_cnt: int,
) -> IntArray:
    adj_list = to_adj_list(graph)
    
    return sample_neighbor_from_adj_list(
        adj_list = adj_list,
        sample_neighbor_cnt = sample_neighbor_cnt, 
    )
