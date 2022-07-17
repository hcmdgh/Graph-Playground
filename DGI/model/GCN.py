from dl import * 


class GCN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int,
                 act: Callable,
                 dropout: float):
        super(GCN, self).__init__()

        assert num_layers >= 2 

        self.gcn_layers = nn.ModuleList([
            dglnn.GraphConv(in_dim, hidden_dim, activation=act),
            *[
                dglnn.GraphConv(hidden_dim, hidden_dim, activation=act)
                for _ in range(num_layers - 2)
            ],
            dglnn.GraphConv(hidden_dim, out_dim), 
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        h = feat
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            if i > 0:
                h = self.dropout(h)
                
            h = gcn_layer(g, h)

        return h
