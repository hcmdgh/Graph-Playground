from util import * 


class SemanticAttention(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 attn_dropout: float = 0.5):
        super().__init__()
        
        self.emb_fc = nn.Linear(emb_dim, emb_dim)
        
        self.attn_fc = nn.Linear(emb_dim, 1, bias=False)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        self.reset_parameters() 
        
        self.device = get_device()
        self.to(self.device)
    
    def reset_parameters(self):
        reset_linear_parameters(self.emb_fc)
        reset_linear_parameters(self.attn_fc)

    # [input]
    #   feat: float[B x num_nodes x emb_dim]
    # [output]
    #   out: float[num_nodes x emb_dim]
    def forward(self, feat: FloatTensor) -> FloatTensor:
        # feat: float[B x num_nodes x emb_dim]
        
        # w: [B x 1]
        w = self.attn_fc(
            torch.mean(  # [B x emb_dim]
                torch.tanh(
                    self.emb_fc(feat)  # [B x num_nodes x emb_dim]
                ),
                dim = 1,
            )
        )
        
        # a: [B x 1 x 1]
        a = torch.softmax(w, dim=0).view(-1, 1, 1)
        
        # out: [num_nodes x emb_dim]
        out = torch.sum(
            a * feat,  # [B x 1 x 1] * [B x num_nodes x emb_dim] -> [B x num_nodes x emb_dim]
            dim = 0,
        )
        
        return out 
