from util import * 
from .han_layer import * 


class HAN(nn.Module):
    def __init__(self,
                 hg: dgl.DGLHeteroGraph,
                 metapaths: list[list[tuple[str, str, str]]],
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int,
                 layer_num_heads: int,
                 dropout_ratio: float = 0.5):
        super().__init__()
        
        self.han_layers = nn.ModuleList()
        self.han_layers.append(HANLayer(
            hg = hg,
            metapaths = metapaths,
            in_dim = in_dim,
            out_dim = hidden_dim,
            layer_num_heads = layer_num_heads,
            dropout_ratio = dropout_ratio, 
        ))
        for _ in range(1, num_layers):
            self.han_layers.append(HANLayer(
                hg = hg,
                metapaths = metapaths,
                in_dim = hidden_dim * layer_num_heads,
                out_dim = hidden_dim,
                layer_num_heads = layer_num_heads,
                dropout_ratio = dropout_ratio, 
            ))
            
        self.predict = nn.Linear(hidden_dim * layer_num_heads, out_dim)
        
    def forward(self, feat: FloatTensor) -> FloatTensor:
        for han_layer in self.han_layers:
            feat = han_layer(feat)
            
        out = self.predict(feat)
        
        return out 
        