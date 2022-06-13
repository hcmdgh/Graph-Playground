from util import * 
from .SAGEConv import * 

__all__ = ['GraphSAGE', 'SAGEConv']


class GraphSAGE(nn.Module):
    def __init__(self,
                 *, 
                 in_dim: Union[int, tuple[int, int]],
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 batch_norm: bool = True, 
                 normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True):
        super().__init__()
        
        self.device = get_device() 
        
        assert num_layers >= 2 
        self.num_layers = num_layers 
        self.batch_norm = batch_norm 
        
        _SAGEConv = functools.partial(
            SAGEConv,
            normalize = normalize,
            root_weight = root_weight,
            bias = bias, 
        ) 

        self.conv_list = nn.ModuleList([
            _SAGEConv(in_dim=in_dim, out_dim=hidden_dim),
            *[
                _SAGEConv(in_dim=hidden_dim, out_dim=hidden_dim)
                for _ in range(num_layers - 2)
            ], 
            _SAGEConv(in_dim=hidden_dim, out_dim=out_dim),
        ])
        
        if batch_norm:
            self.bn_list = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim).to(self.device)
                for _ in range(num_layers - 1)
            ])
        else:
            self.bn_list = [
                lambda x: x
                for _ in range(num_layers - 1)
            ] 
        
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters() 
        
    def reset_parameters(self):
        for conv in self.conv_list:
            conv.reset_parameters() 
            
        if self.batch_norm:
            for bn in self.bn_list:
                bn.reset_parameters() 
        
    def forward(self,
                g: dgl.DGLGraph,
                feat: Union[FloatTensor, tuple[FloatTensor, FloatTensor]]) -> FloatTensor:
        h = feat 
                
        for i in range(self.num_layers - 1):
            conv = self.conv_list[i]
            bn = self.bn_list[i]

            h = conv(g, h)
            h = bn(h)
            h = torch.relu(h)
            h = self.dropout(h)
            
        h = self.conv_list[-1](g, h)
                
        return h 

    
    def forward_batch(self,
                      blocks: list[dgl.DGLGraph],
                      feat: Union[FloatTensor, tuple[FloatTensor, FloatTensor]]) -> FloatTensor:
        h = feat 
                
        for i in range(self.num_layers - 1):
            conv = self.conv_list[i]
            bn = self.bn_list[i]

            h = conv(blocks[i], h)
            h = bn(h)
            h = torch.relu(h)
            h = self.dropout(h)
            
        h = self.conv_list[-1](blocks[-1], h)
                
        return h 
