from dl import * 

__all__ = ['GCN']


class GCN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int = 2,
                 act: Callable = nn.PReLU(),
                 dropout: float = 0.0):
        super().__init__()

        assert num_layers >= 2 
        
        self.layers = nn.ModuleList([
            dglnn.GraphConv(in_dim, hidden_dim, activation=act),
            *[
                self.layers.append(dglnn.GraphConv(hidden_dim, hidden_dim, activation=act))
                for _ in range(num_layers - 2)
            ],
            dglnn.GraphConv(hidden_dim, out_dim),
        ])

        self.dropout = nn.Dropout(dropout)

        self.device = get_device()
        self.to(self.device)

    def forward(self, 
                g: dgl.DGLGraph, 
                feat: FloatTensor) -> FloatTensor:
        h = feat

        for i, layer in enumerate(self.layers):
            if i > 0:
                h = self.dropout(h)

            h = layer(g, h)

        return h 
