from util import * 


class GATConv(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int,
                 dropout_ratio: float = 0.0,
                 negative_slope: float = 0.2,
                 residual: bool = False,
                 activation: Optional[Callable] = None,
                 allow_zero_in_degree: bool = False,
                 bias: bool = True):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.num_heads = num_heads
        self.allow_zero_in_degree = allow_zero_in_degree

        self.W_fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        self.alpha_l = Parameter(torch.zeros(num_heads, out_dim))
        self.alpha_r = Parameter(torch.zeros(num_heads, out_dim))
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self.feat_dropout = nn.Dropout(dropout_ratio)
        self.attn_dropout = nn.Dropout(dropout_ratio)
        
        if bias:
            self.bias = Parameter(torch.zeros(1, num_heads, out_dim))
        else:
            self.bias = None 
            
        if residual:
            if in_dim != out_dim * num_heads:
                self.residual_fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            else:
                self.residual_fc = nn.Identity() 
        else:
            self.residual_fc = None 
        
        self.activation = activation 
        
        self.reset_parameters() 
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        
        nn.init.xavier_normal_(self.W_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.alpha_l, gain=gain)
        nn.init.xavier_normal_(self.alpha_r, gain=gain)
        
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)
            
        if isinstance(self.residual_fc, nn.Linear):
            nn.init.xavier_normal_(self.residual_fc.weight, gain=gain)
    
    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        # feat: [num_nodes x in_dim]
                
        with g.local_scope():
            if not self.allow_zero_in_degree:
                assert not (g.in_degrees() == 0).any()
                
            feat_drop = self.feat_dropout(feat)
            
            # feat_fc: [num_nodes x (num_heads * out_dim)]
            feat_fc = self.W_fc(feat_drop)
            
            # feat_fc: [num_nodes x num_heads x out_dim]
            feat_fc = feat_fc.view(-1, self.num_heads, self.out_dim)

            # attn_l: [num_nodes x num_heads x 1], alpha_l: [num_heads x out_dim]
            # attn_r: [num_nodes x num_heads x 1], alpha_r: [num_heads x out_dim]
            attn_l = torch.einsum('bhd,hd->bh', feat_fc, self.alpha_l).unsqueeze(dim=-1)
            attn_r = torch.einsum('bhd,hd->bh', feat_fc, self.alpha_r).unsqueeze(dim=-1)
            
            g.ndata['_feat_fc'] = feat_fc
            g.ndata['_attn_l'] = attn_l 
            g.ndata['_attn_r'] = attn_r
            
            g.apply_edges(dglfn.u_add_v('_attn_l', '_attn_r', '_attn'))
            
            # edge_attn: [num_edges x num_heads x 1]
            edge_attn = g.edata.pop('_attn') 
            edge_attn = self.leaky_relu(edge_attn)
            edge_attn = dglF.edge_softmax(g, edge_attn)
            edge_attn = self.attn_dropout(edge_attn)
            
            g.edata['_attn'] = edge_attn 
            
            g.update_all(
                message_func = dglfn.u_mul_e('_feat_fc', '_attn', '_'),
                reduce_func = dglfn.sum('_', '_out_feat'),
            )
            
            # out_feat: [num_nodes x num_heads x out_dim]
            out_feat = g.ndata.pop('_out_feat')
            
            if self.residual_fc is not None:
                # feat_drop: [num_nodes x in_dim]
                # residual_val: [num_nodes x num_heads x out_dim]
                residual_val = self.residual_fc(feat_drop).view(-1, self.num_heads, self.out_dim)

                out_feat = out_feat + residual_val 
                
            if self.bias is not None:
                # self.bias: [1 x num_heads x out_dim] 
                out_feat = out_feat + self.bias 
            
            if self.activation is not None:
                out_feat = self.activation(out_feat)
                
            return out_feat 
