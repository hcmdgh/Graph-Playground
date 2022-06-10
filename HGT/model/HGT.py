from util import * 
from .HGTConv import * 

__all__ = ['HGT', 'HGT_Official']


class HGT(nn.Module):
    def __init__(self,
                 *,
                 in_dim: Union[int, dict[str, int]],
                 hidden_dim: int = 64, 
                 out_dim: int,
                 node_types: set[str],
                 edge_types: set[tuple[str, str, str]],
                 num_heads: int = 2,
                 num_layers: int = 1):
        super().__init__()
        
        if isinstance(in_dim, int):
            self.in_dim_dict = {node_type: in_dim for node_type in node_types}
        elif isinstance(in_dim, dict):
            assert set(in_dim.keys()) == node_types 
            self.in_dim_dict = in_dim 
        else:
            raise AssertionError 
        
        self.in_fc_dict = nn.ModuleDict({
            node_type: nn.Linear(_in_dim, hidden_dim)
            for node_type, _in_dim in self.in_dim_dict.items()
        })
        
        self.hgt_conv_list = nn.ModuleList([
            HGTConv(
                in_dim = hidden_dim,
                out_dim = hidden_dim,
                node_types = node_types,
                edge_types = edge_types,
                num_heads = num_heads, 
            )
            for _ in range(num_layers)
        ])
        
        self.out_fc_dict = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, out_dim)
            for node_type in node_types
        })
        
        self.device = get_device() 
        self.to(self.device)
        
    def forward(self,
                hg: dgl.DGLHeteroGraph,
                feat_dict: dict[str, FloatTensor]) -> dict[str, FloatTensor]:
        hidden_dict = {
            node_type: torch.relu(
                self.in_fc_dict[node_type](feat)
            )
            for node_type, feat in feat_dict.items() 
        } 
        
        for hgt_conv in self.hgt_conv_list:
            hidden_dict = hgt_conv(hg=hg, feat_dict=hidden_dict)

        out_dict = {
            node_type: self.out_fc_dict[node_type](feat)
            for node_type, feat in hidden_dict.items() 
        } 
            
        return out_dict


class HGT_Official(nn.Module):
    def __init__(self,
                 *,
                 in_dim: Union[int, dict[str, int]],
                 hidden_dim: int = 64, 
                 out_dim: int,
                 node_types: set[str],
                 edge_types: set[tuple[str, str, str]],
                 num_heads: int = 2,
                 num_layers: int = 1):
        super().__init__()
        
        if isinstance(in_dim, int):
            self.in_dim_dict = {node_type: in_dim for node_type in node_types}
        elif isinstance(in_dim, dict):
            assert set(in_dim.keys()) == node_types 
            self.in_dim_dict = in_dim 
        else:
            raise AssertionError 
        
        self.in_fc_dict = nn.ModuleDict({
            node_type: nn.Linear(_in_dim, hidden_dim)
            for node_type, _in_dim in self.in_dim_dict.items()
        })
        
        self.hgt_conv_list = nn.ModuleList([
            pygnn.HGTConv(
                hidden_dim, 
                hidden_dim, 
                metadata = (list(node_types), list(edge_types)),
                heads = num_heads, 
                group = 'mean',
            )   
            for _ in range(num_layers)
        ])
        
        self.out_fc_dict = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, out_dim)
            for node_type in node_types
        })
        
        self.device = get_device() 
        self.to(self.device)
        
    def forward(self,
                hg: dgl.DGLHeteroGraph,
                feat_dict: dict[str, FloatTensor]) -> dict[str, FloatTensor]:
        hidden_dict = {
            node_type: torch.relu(
                self.in_fc_dict[node_type](feat)
            )
            for node_type, feat in feat_dict.items() 
        } 
        
        edge_index_dict = get_dgl_hg_edge_index_dict(hg=hg, return_type='pyg')
        
        for hgt_conv in self.hgt_conv_list:
            hidden_dict = hgt_conv(hidden_dict, edge_index_dict)

        out_dict = {
            node_type: self.out_fc_dict[node_type](feat)
            for node_type, feat in hidden_dict.items() 
        } 
            
        return out_dict
