from .imports import * 
from sklearn.decomposition import PCA

__all__ = ['perform_PCA']


def perform_PCA(feat: FloatArray,
                out_dim: int) -> FloatArray:
    out = PCA(n_components=out_dim).fit_transform(feat)
    
    return out 
