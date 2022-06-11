from util import * 


class GraphConv(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 weight: bool = True,
                 bias: bool = True,
                 activation: Optional[Callable] = None,
                 allow_zero_in_degree: bool = False):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.allow_zero_in_degree = allow_zero_in_degree
        self.activation = activation
        
        if weight:
            self.weight = Parameter(torch.zeros(in_dim, out_dim))
        else:
            self.weight = None 
            
        if bias:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.bias = None 
        
        self.reset_parameters() 
        
    def reset_parameters(self):
        if self.weight is not None:
            glorot_(self.weight)
            
        if self.bias is not None:
            zeros_(self.bias)
            
    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        if not self.allow_zero_in_degree:
            assert not (g.in_degrees() == 0).any()
        
        # 支持二分图
        # feat_D无需使用
        feat_S = feat_D = feat 
        
        # norm: [num_nodes_S]
        out_degrees = g.out_degrees().float().clamp(min=1.)
        norm = torch.pow(out_degrees, -0.5)
        
        # norm: [num_nodes_S x 1]
        norm = norm.view(-1, 1)
        
        feat_S = feat_S * norm 
        
        if self.in_dim > self.out_dim:
            # 先维度变换，再消息传递
            
            if self.weight is not None:
                # [num_nodes_S x out_dim] = [num_nodes_S x in_dim] @ [in_dim x out_dim]
                feat_S = feat_S @ self.weight 
            
            # feat_S: [num_nodes_S x out_dim]    
            g.srcdata['_feat'] = feat_S 
            
            g.update_all(
                message_func = dglfn.copy_src('_feat', '_'),
                reduce_func = dglfn.sum('_', '_out'),
            )
            
            g.srcdata.pop('_feat')
            
            # out_T: [num_nodes_T x out_dim]
            out_T = g.dstdata.pop('_out')

        else:
            # 先消息传递，再维度变换
            
            # feat_S: [num_nodes_S x in_dim]    
            g.srcdata['_feat'] = feat_S 
            
            g.update_all(
                message_func = dglfn.copy_src('_feat', '_'),
                reduce_func = dglfn.sum('_', '_out'),
            )
            
            g.srcdata.pop('_feat')
            
            # out_T: [num_nodes_T x in_dim]
            out_T = g.dstdata.pop('_out')
            
            if self.weight is not None:
                # [num_nodes_T x out_dim] = [num_nodes_T x in_dim] @ [in_dim x out_dim]
                out_T = out_T @ self.weight 
                
        # norm: [num_nodes_T]
        in_degrees = g.in_degrees().float().clamp(min=1.)
        norm = torch.pow(in_degrees, -0.5)

        # norm: [num_nodes_T x 1]
        norm = norm.reshape(-1, 1)
        
        out_T = out_T * norm 
        
        if self.bias is not None:
            out_T = out_T + self.bias 
            
        if self.activation is not None:
            out_T = self.activation(out_T)
            
        return out_T 
