from util import * 
from .sage_conv import * 


class OfficialSAGEConv(dglnn.SAGEConv):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 aggr_type: Literal['mean', 'gcn', 'pool', 'lstm'],
                 dropout: float = 0.0,
                 bias: bool = True,
                 norm: Optional[Callable] = None,
                 act: Optional[Callable] = None):
        super().__init__(
            in_feats = in_dim,
            out_feats = out_dim,
            aggregator_type = aggr_type,
            feat_drop = dropout,
            bias = bias,
            norm = norm, 
            activation = act, 
        )
        

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int,
                 aggr_type: Literal['mean', 'gcn', 'pool', 'lstm'],
                 act: Callable = torch.relu,
                 dropout: float = 0.0,
                 official: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.num_layers = num_layers

        # [BEGIN] SAGEConv Layers
        self.layers = nn.ModuleList()
        
        if official:
            _SAGEConv = OfficialSAGEConv 
        else:
            _SAGEConv = SAGEConv
        
        self.layers.append(_SAGEConv(
            in_dim = in_dim,
            out_dim = hidden_dim,
            aggr_type = aggr_type,
        ))

        for _ in range(num_layers - 1):
            self.layers.append(_SAGEConv(
                in_dim = hidden_dim,
                out_dim = hidden_dim,
                aggr_type = aggr_type,
            ))
            
        self.layers.append(_SAGEConv(
            in_dim = hidden_dim,
            out_dim = out_dim,
            aggr_type = aggr_type,
        ))
        # [END]

    def forward(self, 
                graph: dgl.DGLGraph, 
                feat: FloatTensor):
        h = self.dropout(feat)

        for l, layer in enumerate(self.layers):
            h = layer(graph, h)

            if l < len(self.layers) - 1:
                h = self.act(h)
                h = self.dropout(h)

        return h
