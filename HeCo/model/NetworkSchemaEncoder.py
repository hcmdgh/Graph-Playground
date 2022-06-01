from util import * 
from .HeCoGATConv import * 
from .SemanticAttention import * 
from dgl.sampling import sample_neighbors


class NetworkSchemaEncoder(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 neighbor_size_list: list[int],
                 attn_dropout: float = 0.0):
        super().__init__()
        
        self.neighbor_size_list = neighbor_size_list 
        
        self.heco_gats = nn.ModuleList([
            HeCoGATConv(
                emb_dim = emb_dim,
                attn_dropout = attn_dropout,
                activation = F.elu, 
            )
            for _ in neighbor_size_list
        ])
        
        self.semantic_attn = SemanticAttention(
            emb_dim = emb_dim,
            attn_dropout = attn_dropout,
        )
        
    # [input]
    #   bg_list: bg[num_metapaths]
    #   feat_S_list: float[num_metapaths x num_nodes_S x emb_dim]
    #   feat_T: float[num_nodes_T x emb_dim]
    # [output]
    #   out: float[num_nodes_T x emb_dim]
    def forward(self,
                bg_list: list[dgl.DGLHeteroGraph],
                feat_S_list: list[FloatTensor],
                feat_T: FloatTensor) -> FloatTensor:
        # bg_list: bg[num_metapaths]
        # feat_S_list: float[num_metapaths x num_nodes_S x emb_dim]
        # feat_T: float[num_nodes_T x emb_dim]
        
        assert len(feat_S_list) == len(self.neighbor_size_list) == len(bg_list) == len(self.heco_gats)
                
        gat_emb_list = []
        
        for neighbor_size, bg, feat_S, heco_gat in zip(self.neighbor_size_list, bg_list, feat_S_list, self.heco_gats):
            node_dict = {
                bg.dsttypes[0]: bg.dstnodes()
            }
            
            sampled_graph = to_device(sample_neighbors(
                g = bg, 
                nodes = node_dict, 
                fanout = neighbor_size,
            )) 
            
            # gat_emb: [num_nodes_T x emb_dim]
            gat_emb = heco_gat(
                bg = sampled_graph,
                feat_S = feat_S,
                feat_T = feat_T,
            )
            
            gat_emb_list.append(gat_emb)
            
        # combined_gat_emb: [num_nodes_T x num_metapaths x emb_dim]
        combined_gat_emb = torch.stack(gat_emb_list, dim=1)
        
        # out: [num_nodes_T x emb_dim]
        out = self.semantic_attn(combined_gat_emb)
        
        return out 
