from dl import * 


class GraphEncoder(nn.Module):
    def __init__(self,
                 *,
                 in_dim: int,
                 hidden_dim: Optional[int] = None,
                 out_dim: int,
                 activation: Callable = torch.relu,
                 num_layers: int = 2):
        super().__init__()
        
        assert num_layers >= 2
        
        if hidden_dim is None:
            hidden_dim = out_dim * 2 
        
        self.conv_list = nn.ModuleList([
            dglnn.GraphConv(in_dim, hidden_dim),
            *[
                dglnn.GraphConv(hidden_dim, hidden_dim)
                for _ in range(num_layers - 2)
            ],
            dglnn.GraphConv(hidden_dim, out_dim),
        ])
        
        self.activation = activation
        
        self.device = get_device()
        self.to(self.device)
        
    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        h = feat 
                
        for conv in self.conv_list:
            h = self.activation(
                conv(g, h)
            )
            
        return h 
