from util import *


def combine_graphs(src_g: dgl.DGLGraph,
                   tgt_g: dgl.DGLGraph,
                   add_super_node: bool) -> dgl.DGLGraph:
    """
    合并两个同构图，并添加超级点。（注：图必须在CPU上）
    """
    src_g.ndata['origin'] = torch.full(size=[src_g.num_nodes()], fill_value=1, dtype=torch.int64)
    tgt_g.ndata['origin'] = torch.full(size=[tgt_g.num_nodes()], fill_value=2, dtype=torch.int64)

    combined_g = dgl.batch([src_g, tgt_g])
    
    # [BEGIN] 添加超级点
    num_nodes = combined_g.num_nodes() 
    nids = torch.arange(num_nodes)
    origin = combined_g.ndata['origin']
    src_nids = nids[origin == 1]
    tgt_nids = nids[origin == 2]
    
    combined_g = dgl.add_nodes(g=combined_g, num=2)

    if add_super_node:
        combined_g = dgl.add_edges(g=combined_g,
                                u=torch.tensor([num_nodes] * len(src_nids), dtype=torch.int64),
                                v=src_nids)

        combined_g = dgl.add_edges(g=combined_g,
                                u=torch.tensor([num_nodes + 1] * len(tgt_nids), dtype=torch.int64),
                                v=tgt_nids)
        
        combined_g = dgl.add_edges(g=combined_g, u=num_nodes, v=num_nodes + 1)
        
        combined_g = dgl.to_bidirected(combined_g, copy_ndata=True)
    # [END]
    
    combined_g = dgl.remove_self_loop(combined_g)
    combined_g = dgl.add_self_loop(combined_g)
    
    return combined_g 


def norm_adj_mat(adj_mat: FloatArray) -> FloatArray:
    row_sum = np.sum(adj_mat, axis=1, keepdims=True)
    row_sum_inv = np.power(row_sum, -1)
    row_sum_inv[np.isinf(row_sum_inv)] = 0.
    
    out_adj_mat = adj_mat * row_sum_inv 
    
    return out_adj_mat 


def aggr_adj_mat(adj_mat: FloatArray,
                 step: int = 3) -> FloatArray:
    assert step >= 2 
                 
    adj_mat = norm_adj_mat(adj_mat)
    a_k = a = adj_mat

    for k in range(2, step + 1):
        a_k = a_k @ adj_mat
        a = a + a_k / k
        
    return a


def calc_PPMI_mat(adj_mat: FloatArray) -> FloatArray:
    np.fill_diagonal(adj_mat, 0.)
    adj_mat = norm_adj_mat(adj_mat)
    N = len(adj_mat)

    col_sum = np.sum(adj_mat, axis=0, keepdims=True)
    col_sum[col_sum == 0.] = 1. 

    ppmi = np.log(N * adj_mat / col_sum)
    ppmi[np.isnan(ppmi)] = 0.
    ppmi[ppmi < 0.] = 0. 
    
    return ppmi
