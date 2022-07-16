from dl import * 


class MLP(nn.Module):
    def __init__(self,
                 *, 
                 in_dim: int,
                 hidden_dim: Optional[int] = None,
                 out_dim: int):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = (in_dim + out_dim) // 2 
        
        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim), 
        )
        
        self.device = get_device()
        self.to(self.device)
        
    def forward(self, inp: FloatTensor) -> FloatTensor:
        return self.seq(inp)
