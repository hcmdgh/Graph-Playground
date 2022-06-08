from util import * 


class FeatureExtractor(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 dropout: float = 0.5):
        super().__init__()
        
        hidden_dim = (in_dim + out_dim) // 2 

        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, inp: FloatTensor) -> FloatTensor:
        out = self.seq(inp)

        return out 
