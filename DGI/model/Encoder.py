from dl import * 

from .GCN import * 


class Encoder(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 emb_dim: int, 
                 num_layers: int, 
                 act: Callable, 
                 dropout: float):
        super().__init__()

        self.gcn = GCN(
            in_dim = in_dim, 
            hidden_dim = emb_dim, 
            out_dim = emb_dim, 
            num_layers = num_layers, 
            act = act, 
            dropout = dropout,
        )

    def forward(self, 
                g: dgl.DGLGraph, 
                feat: FloatTensor, 
                corrupt: bool):
        if corrupt:
            perm = torch.randperm(g.num_nodes())
            feat = feat[perm]
            
        feat = self.gcn(g, feat)

        return feat
