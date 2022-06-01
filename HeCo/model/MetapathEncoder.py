from util import * 
from .SemanticAttention import * 


class MetapathEncoder(nn.Module):
    def __init__(self,
                 num_metapaths: int,
                 emb_dim: int,
                 attn_dropout: float = 0.0):
        super().__init__()
        
        self.gcn_convs = nn.ModuleList([
            dglnn.GraphConv(
                in_feats = emb_dim,
                out_feats = emb_dim,
                norm = 'right',
                activation = nn.PReLU(),
            )
            for _ in range(num_metapaths)
        ])
        
        self.semantic_attn = SemanticAttention(
            emb_dim = emb_dim,
            attn_dropout = attn_dropout,
        )
    
    # [input]
    #   metapath_subgraph_list: g[num_metapaths]
    #   feat: float[num_nodes x emb_dim]      
    # [output]
    #   out: float[num_nodes x emb_dim]
    def forward(self,
                metapath_subgraph_list: list[dgl.DGLGraph],
                feat: FloatTensor):
        # metapath_subgraph_list: g[num_metapaths]
        # feat: float[num_nodes x emb_dim]

        assert len(self.gcn_convs) == len(metapath_subgraph_list)
        
        emb_list = [
            gcn_conv(g, feat)
            for gcn_conv, g in zip(self.gcn_convs, metapath_subgraph_list)
        ]

        # emb: [num_nodes x num_metapaths x emb_dim]
        emb = torch.stack(emb_list, dim=1)
        
        # out: [num_nodes x emb_dim]
        out = self.semantic_attn(emb)
        
        return out 
