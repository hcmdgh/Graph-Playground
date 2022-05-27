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
        
        self.num_heads = num_heads 
        self.in_dim = in_dim
        self.out_dim = out_dim 
        self.allow_zero_in_degree = allow_zero_in_degree
        
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        self.attn_l = Parameter(torch.zeros(1, num_heads, out_dim))
        self.attn_r = Parameter(torch.zeros(1, num_heads, out_dim))

        self.feat_dropout = nn.Dropout(dropout_ratio)
        self.attn_dropout = nn.Dropout(dropout_ratio)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
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
        
        nn.init.xavier_normal_(self.fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        
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
            
            # -> [num_nodes x (num_heads * out_dim)] 
            feat_fc = self.fc(feat_drop)
            
            # -> [num_nodes x num_heads x out_dim]
            feat_fc = feat_fc.view(-1, self.num_heads, self.out_dim)
            
            # el: [num_nodes x num_heads x 1]
            el = torch.unsqueeze(
                # [num_nodes x num_heads]
                torch.sum(
                    # self.attn_l: [1 x num_heads x out_dim]
                    feat_fc * self.attn_l,  
                    dim=-1, 
                ),
                dim=-1, 
            )
            
            # er: [num_nodes x num_heads x 1]
            er = torch.unsqueeze(
                torch.sum(
                    feat_fc * self.attn_r,  
                    dim=-1, 
                ),
                dim=-1, 
            )
            
            g.ndata['ft'] = feat_fc 
            g.ndata['el'] = el 
            g.ndata['er'] = er 
            
            # e: [num_edges x num_heads x 1]
            g.apply_edges(dglfn.u_add_v('el', 'er', 'e'))

            # e: [num_edges x num_heads x 1]
            e = self.leaky_relu(g.edata.pop('e'))
            
            # a: [num_edges x num_heads x 1]
            g.edata['a'] = self.attn_dropout(
                dglF.edge_softmax(g, e)
            )
            
            g.update_all(
                message_func = dglfn.u_mul_e('ft', 'a', 'm'),
                reduce_func = dglfn.sum('m', 'ft'),
            )
            
            # out: [num_nodes x num_heads x out_dim]
            out = g.ndata.pop('ft')
            
            if self.residual_fc is not None:
                # feat_drop: [num_nodes x in_dim]
                # residual_val: [num_nodes x num_heads x out_dim]
                residual_val = self.residual_fc(feat_drop).view(-1, self.num_heads, self.out_dim)

                out = out + residual_val 
                
            if self.bias is not None:
                # self.bias: [1 x num_heads x out_dim] 
                out = out + self.bias 
                
            if self.activation is not None:
                out = self.activation(out)
                
            # out: [num_nodes x num_heads x out_dim]
            return out 
