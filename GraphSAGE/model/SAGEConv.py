from util import * 


class SAGEConv(nn.Module):
    def __init__(self,
                 in_dim: Union[int, tuple[int, int]],
                 out_dim: int,
                 normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True):
        super().__init__()
        
        self.out_dim = out_dim 
        self.normalize = normalize 
        self.root_weight = root_weight 
        self.bias = bias 
        
        if isinstance(in_dim, int):
            self.in_dim_S = self.in_dim_D = in_dim 
        elif isinstance(in_dim, tuple):
            self.in_dim_S, self.in_dim_D = in_dim
        else:
            raise AssertionError 
        
        self.fc_S = nn.Linear(self.in_dim_S, out_dim, bias=bias)
        
        if self.root_weight:
            self.fc_D = nn.Linear(self.in_dim_D, out_dim, bias=False)
            
        self.reset_parameters() 
        
        self.device = get_device() 
        self.to(self.device)
        
    def reset_parameters(self):
        reset_linear_parameters(self.fc_S)
        
        if self.root_weight:
            reset_linear_parameters(self.fc_D)
            
    def forward(self,
                g: dgl.DGLGraph,
                feat: Union[FloatTensor, tuple[FloatTensor, FloatTensor]]) -> FloatTensor:
        if isinstance(feat, FloatTensor):
            feat_S = feat_D = feat
            
            if g.is_block:
                feat_D = feat_S[:g.num_dst_nodes()] 
        elif isinstance(feat, tuple):
            feat_S, feat_D = feat 
        else:
            raise AssertionError 
        
        g.srcdata['_feat'] = feat_S 
        
        g.update_all(
            message_func = dglfn.copy_src('_feat', '_'), 
            reduce_func = dglfn.mean('_', '_out'), 
        )
        
        g.srcdata.pop('_feat')

        # out: [num_nodes_D x in_dim_S]
        out = g.dstdata.pop('_out')
        
        # out: [num_nodes_D x out_dim]
        out = self.fc_S(out)
        
        if self.root_weight and feat_D is not None:
            out += self.fc_D(feat_D)
            
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
            
        return out 
