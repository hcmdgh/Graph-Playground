from util import * 


class DomainDiscriminator(nn.Module):
    def __init__(self, 
                 in_dim: int):
        super().__init__()

        hidden_dim = in_dim // 2 

        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2), 
        )

    def forward(self, inp: FloatTensor) -> FloatTensor:
        out = self.seq(inp)

        return out 
