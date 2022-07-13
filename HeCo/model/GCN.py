from dl import * 


class GCN(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 act: Callable = nn.PReLU(), 
                 bias: bool = True):
        super().__init__()

        self.gcn_layer = dglnn.GraphConv(
            in_feats = in_dim,
            out_feats = out_dim,
            bias = bias,
            activation = act, 
        )

    def forward(self,
                graph: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        out = self.gcn_layer(graph, feat)
        
        return out 
