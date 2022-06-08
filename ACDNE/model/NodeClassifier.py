from util import * 


class NodeClassifier(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 out_dim: int):
        super().__init__()

        hidden_dim = (in_dim + out_dim) // 2 

        self.seq = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, inp: FloatTensor) -> FloatTensor:
        out = self.seq(inp)

        return out 
