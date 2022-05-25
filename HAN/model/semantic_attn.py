from util import * 
from ..config import * 


class SemanticAttention(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
    def forward(self, inp: FloatTensor) -> FloatTensor:
        # inp: float[num_nodes x num_metapaths x in_dim]
        
        # weight: [1 x num_metapaths x 1]
        weight = torch.mean(
            self.project(inp),  # [num_nodes x num_metapaths x 1]
            dim=0,
            keepdim=True,
        ) 
        
        # weight: [1 x num_metapaths x 1]
        if not REMOVE_SOFTMAX_IN_SEMANTIC_ATTN:
            weight = torch.softmax(weight, dim=1) 
            
        if REMOVE_SEMANTIC_ATTN:
            weight = to_device(torch.ones_like(weight))
        
        # out: [num_nodes x in_dim]
        out = torch.sum(
            weight * inp,  # [num_nodes x num_metapaths x in_dim]
            dim=1,
        ) 
        
        return out 
