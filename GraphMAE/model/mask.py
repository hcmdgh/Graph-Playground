from dl import * 

__all__ = ['mask_node_feat']


def mask_node_feat(
    feat: FloatTensor,
    mask_token: FloatTensor,
    mask_ratio: float = 0.3,
) -> tuple[FloatTensor, IntArray]:
    num_nodes = len(feat)
    perm = np.random.permutation(num_nodes)
    
    num_mask_nodes = int(num_nodes * mask_ratio)
    mask_nodes = perm[:num_mask_nodes]
    
    out_feat = feat.clone() 
    
    out_feat[mask_nodes] = mask_token 
    
    return out_feat, mask_nodes 
