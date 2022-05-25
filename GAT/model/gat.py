from util import * 
from .gat_conv import * 


class GAT(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_heads: int,
                 num_layers: int,
                 activation: Optional[Callable] = F.elu,
                 dropout_ratio: float = 0.0,
                 negative_slope: float = 0.2,
                 residual: bool = False):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.activation = activation 
        
        self.gat_layers = nn.ModuleList()
        
        if num_layers == 1:
            self.gat_layers.append(GATConv(
                in_dim = in_dim, 
                out_dim = out_dim, 
                num_heads = num_heads,
                dropout_ratio = dropout_ratio, 
                negative_slope = negative_slope,
                residual = residual, 
                activation = None,
            ))
        
        elif num_layers > 1:
            self.gat_layers.append(GATConv(
                in_dim = in_dim, 
                out_dim = hidden_dim, 
                num_heads = num_heads,
                dropout_ratio = dropout_ratio, 
                negative_slope = negative_slope,
                residual = False, 
                activation = self.activation,
            ))

            for _ in range(num_layers - 2):
                self.gat_layers.append(GATConv(
                    in_dim = hidden_dim * num_heads, 
                    out_dim = hidden_dim, 
                    num_heads = num_heads,
                    dropout_ratio = dropout_ratio, 
                    negative_slope = negative_slope,
                    residual = residual, 
                    activation = self.activation,
                ))

            self.gat_layers.append(GATConv(
                in_dim = hidden_dim * num_heads, 
                out_dim = out_dim, 
                num_heads = num_heads,
                dropout_ratio = dropout_ratio, 
                negative_slope = negative_slope,
                residual = residual, 
                activation = None,
            ))
            
        else:
            raise AssertionError 
        
    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        # feat: [num_nodes x in_dim]
        h = feat 
        
        for l in range(self.num_layers):
            # h: [num_nodes x num_heads x out_dim]
            h = self.gat_layers[l](g, h)

            if l < self.num_layers - 1:
                # h: [num_nodes x (num_heads * out_dim)]
                h = torch.flatten(h, start_dim=1)
            else:
                # h: [num_nodes x out_dim]
                h = torch.mean(h, dim=1)

        # h: [num_nodes x out_dim]
        return h
