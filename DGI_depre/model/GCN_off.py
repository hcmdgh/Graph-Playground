from util import * 


class GCN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 bias: bool = True):
        super().__init__()
        
        self.device = get_device()
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False).to(self.device)
        
        self.activation = nn.PReLU().to(self.device)
        
        if bias:
            self.bias = Parameter(torch.zeros(out_dim)).to(self.device)
        else:
            self.bias = None 
            
    def forward(self,
                adj_mat: FloatTensor,
                feat: FloatTensor) -> FloatTensor:
        h = self.fc(feat)
        
        out = adj_mat @ h  
        
        if self.bias is not None:
            out += self.bias 
            
        out = self.activation(out)
            
        return out 
