from graph import * 
from .GCN import * 
from .Attention import * 

from dl import * 


class MetapathEncoder(nn.Module):
    def __init__(self, 
                 graph: HeCoGraph, 
                 emb_dim: int, 
                 attn_dropout: float):
        super().__init__()

        self.graph = graph 

        self.gcn_list = nn.ModuleList([
            GCN(in_dim=emb_dim, out_dim=emb_dim) 
            for _ in range(len(graph.metapath_list))
        ])

        self.semantic_attn = Attention(emb_dim=emb_dim, attn_dropout=attn_dropout)

    def forward(self, 
                feat: FloatTensor) -> FloatTensor:
        embed_list = []
        
        for i, subgraph in enumerate(self.graph.metapath_subgraph_list):
            emb = self.gcn_list[i](graph=subgraph, feat=feat)
            embed_list.append(emb)

        embed_list = torch.stack(embed_list)
            
        out = self.semantic_attn(embed_list)

        return out 
