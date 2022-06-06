from util import * 


class SAGEConv(pygconv.MessagePassing):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 norm: bool = False,
                 root_weight: bool = True,
                 bias: bool = True):
        super().__init__(aggr='mean')
        
        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.norm = norm 
        self.root_weight = root_weight 
        
        self.fc_L = nn.Linear(in_dim, out_dim, bias=bias)
        
        if self.root_weight:
            self.fc_R = nn.Linear(in_dim, out_dim, bias=False)
            
        self.reset_parameters() 
        
    def reset_parameters(self):
        self.fc_L.reset_parameters() 
        
        if self.root_weight:
            self.fc_R.reset_parameters() 
            
    def forward(self,
                edge_index: IntTensor,
                feat: FloatTensor) -> FloatTensor:
        out = self.propagate(edge_index, feat=feat)
        
        out = self.fc_L(out)
        
        if self.root_weight:
            out += self.fc_R(feat)
            
        if self.norm: 
            out = F.normalize(out, p=2., dim=-1)
            
        return out 
    
    def message(self, feat_j: FloatTensor) -> FloatTensor:
        return feat_j 
