from util import * 
from .SAGEConv import * 

__all__ = ['GraphSAGE', 'SAGEConv']


class GraphSAGE(nn.Module):
    def __init__(self,
                 *, 
                 in_dim: Union[int, tuple[int, int]],
                 out_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True):
        super().__init__()
        
        assert num_layers > 0 

        self.layer_list = nn.ModuleList([
            SAGEConv(
                in_dim = in_dim,
                out_dim = out_dim,
                normalize = normalize,
                root_weight = root_weight,
                bias = bias, 
            ),
            *[
                SAGEConv(
                    in_dim = out_dim,
                    out_dim = out_dim,
                    normalize = normalize,
                    root_weight = root_weight,
                    bias = bias, 
                )
                for _ in range(num_layers - 1)
            ], 
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.device = get_device() 
        
    def forward(self,
                g: dgl.DGLGraph,
                feat: Union[FloatTensor, tuple[FloatTensor, FloatTensor]]) -> FloatTensor:
        h = feat 
                
        for i, layer in enumerate(self.layer_list):
            h = layer(g, h)
            
            if i < len(self.layer_list) - 1:
                h = torch.relu(h)
                h = self.dropout(h)
                
        return h 
