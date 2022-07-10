import torch
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = act 

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ft))
        else:
            self.bias = None 

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)

        if self.bias is not None:
            out += self.bias
        
        return self.act(out), seq_fts
