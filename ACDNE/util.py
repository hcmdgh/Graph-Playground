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
