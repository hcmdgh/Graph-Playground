from util import * 


class SemanticAttention(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 attn_dropout: float = 0.0):
        super().__init__()
        
        self.fc = nn.Linear(emb_dim, emb_dim)
        
        self.attn = nn.Linear(emb_dim, 1, bias=False)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.reset_parameters() 
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain)
        nn.init.xavier_normal_(self.attn.weight, gain)

    # [input]
    #   feat: float[num_nodes x num_metapaths x emb_dim]
    # [output]
    #   out: float[num_nodes x emb_dim]
    def forward(self, feat: FloatTensor) -> FloatTensor:
        # feat: float[num_nodes x num_metapaths x emb_dim]
        
        # w: [num_metapaths x 1]
        w = self.attn(
            torch.mean(  # [num_metapaths x emb_dim]
                torch.tanh(
                    self.fc(feat)  # [num_nodes x num_metapaths x emb_dim]
                ),
                dim = 0,
            )
        )
        
        # a: [1 x num_metapaths x 1]
        a = torch.softmax(w, dim=0).view(1, -1, 1)
        
        # out: [num_nodes x emb_dim]
        out = torch.sum(
            a * feat,  # [1 x num_metapaths x 1] * [num_nodes x num_metapaths x emb_dim]
            dim = 1,
        )
        
        return out 
