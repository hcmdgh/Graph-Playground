from util import * 
from .HeCoGATConv import * 
from .SemanticAttention import * 
from dgl.sampling import sample_neighbors


class NetworkSchemaEncoder(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 relations: set[EdgeType],
                 attn_dropout: float = 0.5):
        super().__init__()
        
        self.gat_dict = nn.ModuleDict({
            '__'.join(edge_type): HeCoGATConv(
                emb_dim = emb_dim,
                attn_dropout = attn_dropout,
                activation = F.elu, 
            )
            for edge_type in relations 
        })
        
        self.semantic_attn = SemanticAttention(
            emb_dim = emb_dim,
            attn_dropout = attn_dropout,
        )
        
        self.device = get_device() 
        
    # [input]
    #   bg_list: bg[B]
    #   feat_S_list: float[B x num_nodes_S x emb_dim]
    #   feat_D_list: float[B x num_nodes_D x emb_dim]
    #   neighbor_size_list: int[B]
    # [output]
    #   out: float[num_nodes_D x emb_dim]
    def forward(self,
                relation_subgraph_dict: dict[EdgeType, dgl.DGLHeteroGraph],
                relation_neighbor_size_dict: dict[EdgeType, int],
                feat_dict: dict[NodeType, FloatTensor]) -> FloatTensor:
        # bg_list: bg[B]
        # feat_S_list: float[B x num_nodes_S x emb_dim]
        # feat_D_list: float[B x num_nodes_D x emb_dim]
        # neighbor_size_list: int[B]
        
        assert len(relation_subgraph_dict) == len(relation_neighbor_size_dict) == len(self.gat_dict)
                
        gat_emb_list = []
        
        for relation in relation_subgraph_dict:
            bg = relation_subgraph_dict[relation]
            neighbor_size = relation_neighbor_size_dict[relation]
            gat = self.gat_dict['__'.join(relation)]
            feat_S = feat_dict[bg.srctypes[0]]
            feat_D = feat_dict[bg.dsttypes[0]]
            
            # 目标结点类型 -> 目标结点下标Tensor
            node_dict = {
                bg.dsttypes[0]: bg.dstnodes()
            }
            
            sampled_subgraph = sample_neighbors(
                g = bg, 
                nodes = node_dict, 
                fanout = neighbor_size,
            ).to(self.device)
            
            # gat_emb: [num_nodes_D x emb_dim]
            gat_emb = gat(
                bg = sampled_subgraph,
                feat_S = feat_S,
                feat_D = feat_D,
            )
            
            gat_emb_list.append(gat_emb)
            
        # stacked_gat_emb: [B x num_nodes_D x emb_dim]
        stacked_gat_emb = torch.stack(gat_emb_list)
        
        # out: [num_nodes_D x emb_dim]
        out = self.semantic_attn(stacked_gat_emb)
        
        return out 
