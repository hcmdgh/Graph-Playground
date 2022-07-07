from dl import * 


def normalize_feature(feat: FloatTensor) -> FloatTensor:
    row_sum = feat.sum(dim=-1, keepdim=True)
    
    r_inv = row_sum.pow(-1)
    r_inv[torch.isinf(r_inv)] = 0. 
    
    feat = feat * r_inv 
    
    return feat 


def normalize_adj_mat(adj_mat: FloatArray) -> FloatArray:
    d = adj_mat.sum(-1)
    
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    diag = np.diag(d_inv_sqrt) 
    
    # out = diag.T @ adj_mat @ diag 
    out = (adj_mat @ diag).T @ diag 
    
    return out 
