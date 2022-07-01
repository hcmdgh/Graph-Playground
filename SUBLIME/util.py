from util import * 
from sklearn.neighbors import kneighbors_graph

EPS = 1e-10


def mask_feat(feat: FloatTensor,
              mask_ratio: float = 0.6) -> FloatTensor:
    feat = feat
    feat_dim = feat.shape[-1]
    masked_idxs = np.random.choice(feat_dim, size=int(feat_dim * mask_ratio), replace=False)

    mask = torch.ones_like(feat)
    mask[:, masked_idxs] = 0. 
    
    # feat[:, masked_idxs] = 0. 
    feat = feat * mask 
    
    return feat 


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def normalize_adj_mat(adj_mat: FloatTensor, 
                      mode: str = 'sym') -> FloatTensor:
    if mode == "sym":
        inv_sqrt_degree = 1. / (torch.sqrt(adj_mat.sum(dim=-1)) + EPS)
        out = inv_sqrt_degree[:, None] * adj_mat * inv_sqrt_degree[None, :]
    elif mode == "row":
        inv_degree = 1. / (adj_mat.sum(dim=1, keepdim=False) + EPS)
        out = inv_degree[:, None] * adj_mat
    else:
        raise AssertionError 

    return out 


def symmetrize_adj_mat(adj_mat: FloatTensor) -> FloatTensor:
    out = (adj_mat + adj_mat.T) / 2. 
    
    return out 
