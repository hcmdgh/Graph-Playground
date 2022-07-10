import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling
from dl import * 


class GCNEncoder(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int):
        super().__init__()
        
        self.gcn_conv = dglnn.GraphConv(in_feats=in_dim, out_feats=out_dim) 

        # TODO PReLU 参数修改
        self.act = nn.PReLU(out_dim)

    def forward(self,
                graph: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        h = self.gcn_conv(graph, feat)
        h = self.act(h)

        return h 
  
        
class Pool(nn.Module):
    def __init__(self, in_channels, ratio=1.0):
        super(Pool, self).__init__()
        self.sag_pool = SAGPooling(in_channels, ratio)
        
    def forward(self, x, edge, batch, type='mean_pool'):
        if type == 'mean_pool':
            return global_mean_pool(x, batch)
        elif type == 'max_pool':
            return global_max_pool(x, batch)
        elif type == 'sum_pool':
            return global_add_pool(x, batch)
        elif type == 'sag_pool':
            x1, _, _, batch, _, _ = self.sag_pool(x, edge, batch=batch)
            return global_mean_pool(x1, batch)
            