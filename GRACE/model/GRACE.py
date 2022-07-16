from .loss import *
from .MLP import * 
from .GraphEncoder import * 

from dl import * 

__all__ = ['GRACE']


class GRACE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 emb_dim: int = 128,
                 num_gnn_layers: int = 2,
                 tau: float = 0.4):
        super().__init__()
        
        self.tau = tau 
        
        self.mlp = MLP(in_dim=emb_dim, out_dim=emb_dim)

        self.graph_encoder = GraphEncoder(
            in_dim = in_dim,
            out_dim = emb_dim,
            num_layers = num_gnn_layers,
        )
        
        self.device = get_device()
        
    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        out = self.graph_encoder(g=g, feat=feat)
        
        return out 
        
    def calc_contrastive_loss(self,
                              z1: FloatTensor,
                              z2: FloatTensor) -> FloatScalarTensor:
        h1 = self.mlp(z1)
        h2 = self.mlp(z2)

        l1 = calc_pairwise_loss(z1=h1, z2=h2, tau=self.tau)
        l2 = calc_pairwise_loss(z1=h2, z2=h1, tau=self.tau)
        
        res = torch.mean((l1 + l2) / 2) 
        
        return res 
