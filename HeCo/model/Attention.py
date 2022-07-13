from dl import * 


class Attention(nn.Module):
    def __init__(self, 
                 emb_dim: int, 
                 attn_dropout: float):
        super().__init__()
        
        self.in_fc = nn.Linear(emb_dim, emb_dim, bias=True)

        self.attn_fc = nn.Linear(emb_dim, 1, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)

        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.in_fc.weight, gain=1.414)
        nn.init.zeros_(self.in_fc.bias)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=1.414)

    def forward(self, 
                emb_list: FloatTensor) -> FloatTensor:
        # emb_list: [B x num_nodes x emb_dim]

        # _emb_list: [B x emb_dim]
        _emb_list = torch.tanh(self.in_fc(emb_list)).mean(dim=1)
        
        # TODO dropout
        # self.attn_fc.weight = self.attn_dropout(self.attn_fc.weight)
        
        # beta: [B]
        beta = self.attn_fc(_emb_list).squeeze() 

        beta = torch.softmax(beta, dim=0)
        
        # out: [B x num_nodes x emb_dim]
        out = beta.view(-1, 1, 1) * emb_list
        
        # out: [num_nodes x emb_dim]
        out = out.sum(dim=0)
        
        return out 
