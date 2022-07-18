from .GCN import * 

from dl import * 


class Encoder(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 emb_dim: int, 
                 gnn_act: Callable, 
                 gnn_num_layers: int = 2):
        super().__init__()

        self.gnn = GCN(
            in_dim = in_dim,
            hidden_dim = emb_dim,
            out_dim = emb_dim,
            num_layers = gnn_num_layers,
            act = gnn_act,
            dropout = 0.,
        )

    def forward(self, 
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        out = self.gnn(g=g, feat=feat)
        
        return out 
