from util import * 
from .semantic_attn import * 


class HANLayer(nn.Module):
    def __init__(self,
                 hg: dgl.DGLHeteroGraph,
                 metapaths: list[list[tuple[str, str, str]]],
                 in_dim: int,
                 out_dim: int,
                 layer_num_heads: int,
                 dropout_ratio: float = 0.5):
        super().__init__()
        
        self.hg = hg 
        
        self.metapaths = metapaths
        
        self.subgraph_dict = {
            str(metapath): dgl.metapath_reachable_graph(g=hg, metapath=metapath)
            for metapath in metapaths
        }
        
        self.gat_layer_dict = nn.ModuleDict({
            str(metapath): dglnn.GATConv(
                in_feats=in_dim,
                out_feats=out_dim,
                num_heads=layer_num_heads,
                feat_drop=dropout_ratio,
                attn_drop=dropout_ratio,
                activation=F.elu,
                allow_zero_in_degree=True,
            )
            for metapath in metapaths
        })
        
        self.semantic_attn = SemanticAttention(in_dim=out_dim * layer_num_heads)
        
    def forward(self, feat: FloatTensor) -> FloatTensor:
        # feat: float[num_nodes x in_dim]
        
        metapath_emb_list = []
        
        for metapath in self.metapaths:
            subgraph = self.subgraph_dict[str(metapath)]
            
            # gat_out: [num_nodes x num_heads x out_dim]
            gat_out = self.gat_layer_dict[str(metapath)](subgraph, feat)

            # gat_out: [num_nodes x (num_heads * out_dim)]
            gat_out = gat_out.reshape(gat_out.shape[0], -1)
            
            metapath_emb_list.append(gat_out)
            
        # metapath_embs: [num_nodes x num_metapaths x (num_heads * out_dim)]
        metapath_embs = torch.stack(metapath_emb_list, dim=1)
        
        # out: [num_nodes x (num_heads * out_dim)]
        out = self.semantic_attn(metapath_embs)
        
        return out 
