from util import * 


class HeCoGATConv(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 attn_dropout: float = 0.5,
                 negative_slope: float = 0.01,
                 activation: Optional[Callable] = F.elu):
        super().__init__()
        
        self.attn_S = Parameter(torch.zeros(1, emb_dim))
        self.attn_D = Parameter(torch.zeros(1, emb_dim))

        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self.activation = activation 
        
        self.reset_parameters() 
        
        self.device = get_device() 
        self.to(self.device)
        
    def reset_parameters(self):
        glorot_(self.attn_S)
        glorot_(self.attn_D)

    # [input]
    #   bg: 二分图（邻居结点->目标结点）
    #   feat_S: float[num_nodes_S x emb_dim]
    #   feat_D: float[num_nodes_D x emb_dim]
    # [output]
    #   out_D: float[num_nodes_D x emb_dim]
    def forward(self,
                bg: dgl.DGLHeteroGraph,
                feat_S: FloatTensor,
                feat_D: FloatTensor) -> FloatTensor:
        # bg: 二分图（邻居->目标结点）
        # feat_S: float[num_nodes_S x emb_dim]
        # feat_D: float[num_nodes_D x emb_dim]
        
        # attn_S/attn_D: [1 x emb_dim]
        attn_S = self.attn_dropout(self.attn_S)
        attn_D = self.attn_dropout(self.attn_D)
        
        # e_S: [num_nodes_S x 1]
        e_S = (feat_S * attn_S).sum(dim=-1, keepdim=True)

        # e_D: [num_nodes_D x 1]
        e_D = (feat_D * attn_D).sum(dim=-1, keepdim=True)

        # bg.srcdata['_feat_S'] = feat_S 
        bg.srcdata['_e_S'] = e_S 
        bg.dstdata['_e_D'] = e_D 
        
        bg.apply_edges(dglfn.u_add_v('_e_S', '_e_D', '_e'))
        
        bg.srcdata.pop('_e_S') 
        bg.dstdata.pop('_e_D')
        
        # e: [num_edges x 1]
        e = self.leaky_relu(bg.edata.pop('_e')) 

        # a: [num_edges x 1]
        a = dglF.edge_softmax(bg, e)
        bg.edata['_a'] = a 
        
        # feat_S: [num_nodes_S x emb_dim]
        bg.srcdata['_feat_S'] = feat_S

        bg.update_all(
            message_func = dglfn.u_mul_e('_feat_S', '_a', '_'),
            reduce_func = dglfn.sum('_', '_out')
        )
        
        bg.srcdata.pop('_feat_S')
        bg.edata.pop('_a')
        
        # out_D: [num_nodes_D x emb_dim]
        out_D = bg.dstdata.pop('_out')
        
        if self.activation is not None:
            out_D = self.activation(out_D)

        return out_D 
