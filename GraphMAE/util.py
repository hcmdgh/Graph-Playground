from config import * 

from dl import * 

__all__ = [
    'calc_SCE_loss',
    'corrupt_node_feat', 
]


def calc_SCE_loss(v1: FloatTensor,
                  v2: FloatTensor,
                  alpha: float = config.SCE_alpha) -> FloatScalarTensor:
    v1 = F.normalize(v1, p=2, dim=-1)
    v2 = F.normalize(v2, p=2, dim=-1)

    loss = 1 - (v1 * v2).sum(dim=-1)
    loss = loss.pow_(alpha)
    loss = loss.mean() 
    
    return loss 


def corrupt_node_feat(g: dgl.DGLGraph, 
                      feat: FloatTensor, 
                      mask_token: FloatTensor,
                      mask_ratio: float = config.mask_ratio,
                      noise_ratio: float = config.noise_ratio) -> tuple[FloatTensor, IntArray]:
    num_nodes = g.num_nodes()
    num_mask_nodes = int(mask_ratio * num_nodes)
    perm = np.random.permutation(num_nodes)
    mask_nids = perm[:num_mask_nodes]

    feat = feat.clone() 

    if noise_ratio > 0:
        num_noise_nodes = int(noise_ratio * num_mask_nodes)
        _perm = np.random.permutation(num_mask_nodes)
        noise_nids = mask_nids[_perm[:num_noise_nodes]]
        token_nids = mask_nids[_perm[num_noise_nodes:]]
        random_nids = np.random.permutation(num_nodes)[:num_noise_nodes]

        feat[token_nids] = mask_token
        feat[noise_nids] = feat[random_nids]
    else:
        feat[mask_nids] = mask_token

    return feat, mask_nids
