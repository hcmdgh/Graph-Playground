from util import * 


class Discriminator(nn.Module):
    def __init__(self,
                 in_dim: int):
        super().__init__()

        self.in_dim = in_dim 

        self.seq = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, inp: FloatTensor) -> FloatTensor:
        logits = self.seq(inp).squeeze(dim=-1)

        return logits 
