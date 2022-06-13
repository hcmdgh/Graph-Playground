from util import * 
from .GraphConv import * 

__all__ = ['GCN', 'GraphConv']


class GCN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int,
                 activation: Callable = torch.relu,
                 dropout: float = 0.5):
        super().__init__()
        
        assert num_layers >= 2 
        
        self.gcn_layers = nn.ModuleList([
            GraphConv(
                in_dim = in_dim,
                out_dim = hidden_dim,
                activation = activation,
            ),
            *[
                GraphConv(
                    in_dim = hidden_dim,
                    out_dim = hidden_dim,
                    activation = activation,
                ) 
                for _ in range(num_layers - 2)   
            ],
            GraphConv(
                in_dim = hidden_dim,
                out_dim = out_dim,
                activation = None,
            ),
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.device = get_device() 
        self.to(self.device)
        
    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        h = feat 
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            if i > 0:
                h = self.dropout(h)
                
            h = gcn_layer(g, h)
            
        return h 
