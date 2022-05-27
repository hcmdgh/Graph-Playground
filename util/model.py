from .imports import * 


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_layers: int = 2):
        super().__init__()
        
        assert num_layers > 0 
        
        dim_arr = np.linspace(in_dim, out_dim, num_layers + 1, dtype=np.int64)
        
        models = [] 
        
        for i in range(len(dim_arr) - 2):
            models.append(nn.Linear(dim_arr[i], dim_arr[i+1]))
            models.append(nn.ReLU())

        models.append(nn.Linear(dim_arr[-2], dim_arr[-1]))
            
        self.seq = nn.Sequential(*models)
        
    def forward(self, inp: FloatTensor) -> FloatTensor:
        return self.seq(inp)
