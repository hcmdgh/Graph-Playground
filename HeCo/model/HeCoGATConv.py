from util import * 


class HeCoGATConv(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 attn_dropout: float = 0.0,
                 negative_slope: float = 0.01,
                 activation: Optional[Callable] = F.elu):
        super().__init__()
        
        self.attn_l = nn.Linear(emb_dim, 1, bias=False)
        self.attn_r = nn.Linear(emb_dim, 1, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        self.activation = activation 
        
        self.reset_parameters() 
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l.weight, gain)
        nn.init.xavier_normal_(self.attn_r.weight, gain)

    # [input]
    #   bg: 二分图（邻居结点->目标结点）
    #   feat_S: float[num_nodes_S x emb_dim]
    #   feat_T: float[num_nodes_T x emb_dim]
    # [output]
    #   out_feat_T: float[num_nodes_T x emb_dim]
    def forward(self,
                bg: dgl.DGLHeteroGraph,
                feat_S: FloatTensor,
                feat_T: FloatTensor) -> FloatTensor:
        # bg: 二分图（邻居->目标结点）
        # feat_S: float[num_nodes_S x emb_dim]
        # feat_T: float[num_nodes_T x emb_dim]
        
        with bg.local_scope():
            # TODO attn_dropout 
            
            # el: [num_nodes_S x 1]
            el = self.attn_l(feat_S)
            
            # er: [num_nodes_T x 1]
            er = self.attn_r(feat_T)
            
            bg.srcdata['_feat'] = feat_S 
            bg.srcdata['_el'] = el 
            bg.dstdata['_er'] = er 
            
            bg.apply_edges(dglfn.u_add_v('_el', '_er', '_e'))
            
            # e: [num_edges x 1]
            e = bg.edata.pop('_e')
            e = self.leaky_relu(e)
            
            # attn: [num_edges x 1]
            attn = dglF.edge_softmax(graph=bg, logits=e)
            bg.edata['_attn'] = attn 
            
            bg.update_all(message_func=dglfn.u_mul_e('_feat', '_attn', '_'),
                          reduce_func=dglfn.sum('_', '_feat'))
            
            # out_feat_T: [num_nodes_T x emb_dim]
            out_feat_T = bg.dstdata.pop('_feat')
            
            if self.activation is not None:
                out_feat_T = self.activation(out_feat_T)

            return out_feat_T
