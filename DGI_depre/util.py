from util import * 


def normalize_feature(feat: FloatArray) -> FloatArray:
    assert isinstance(feat, FloatArray)
    
    row_sum = np.sum(feat, axis=-1, keepdims=True)
    r_inv = np.power(row_sum, -1)
    r_inv[np.isinf(r_inv)] = 0.

    out = feat * r_inv 

    return out  


def normalize_adj_mat(adj_mat: FloatArray) -> FloatArray:
    assert isinstance(adj_mat, FloatArray)
    
    row_sum = np.sum(adj_mat, axis=-1)
    d_inv_sqrt = np.power(row_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.

    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    out = (adj_mat @ d_mat_inv_sqrt).T @ d_mat_inv_sqrt

    return out 
