from util import * 
from .MetapathEncoder import * 
from .NetworkSchemaEncoder import * 
from .ContrastiveLoss import * 

__all__ = ['HeCo']


class HeCo(nn.Module):
    def __init__(self,
                 hg: dgl.DGLHeteroGraph, 
                 infer_node_type: str, 
                 feat_dict: dict[NodeType, FloatTensor], 
                 metapaths: list[list],
                 relation_neighbor_size_dict: dict[EdgeType, int],
                 positive_sample_mask: BoolArray, 
                 emb_dim: int,
                 tau: float = 0.8,
                 lambda_: float = 0.5,
                 feat_dropout: float = 0.3,
                 attn_dropout: float = 0.5,):
        super().__init__()
        
        self.device = get_device()
        
        self.hg = hg = hg.to(self.device)
        self.infer_node_type = infer_node_type
        self.relation_neighbor_size_dict = relation_neighbor_size_dict
        self.feat_dict = feat_dict = { node_type: feat.to(self.device) for node_type, feat in feat_dict.items() }
        self.positive_sample_mask = positive_sample_mask

        self.in_fc_dict = nn.ModuleDict({
            node_type: nn.Linear(feat.shape[-1], emb_dim).to(self.device)
            for node_type, feat in feat_dict.items() 
        })
        
        self.metapath_subgraph_list = [
            dgl.add_self_loop(
                dgl.remove_self_loop(
                    dgl.metapath_reachable_graph(hg, metapath)
                )
            ).to(self.device)
            for metapath in metapaths
        ]
        
        self.relation_subgraph_dict = {
            edge_type: hg[edge_type]
            for edge_type in relation_neighbor_size_dict
        }
        
        self.feat_dropout = nn.Dropout(feat_dropout)
        
        self.network_schema_encoder = NetworkSchemaEncoder(
            emb_dim = emb_dim,
            relations = set(relation_neighbor_size_dict), 
            attn_dropout = attn_dropout,
        )
        
        self.metapath_encoder = MetapathEncoder(
            num_metapaths = len(metapaths),
            emb_dim = emb_dim,
            attn_dropout = attn_dropout,
        )
        
        self.contrast_model = ContrastiveLoss(
            emb_dim = emb_dim,
            tau = tau,
            lambda_ = lambda_,
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        deep_reset_parameters(self.in_fc_dict)
        
    def forward(self) -> FloatScalarTensor:
        h_dict = {
            node_type: F.elu(
                self.feat_dropout(
                    in_fc(self.feat_dict[node_type])
                )
            )
            for node_type, in_fc in self.in_fc_dict.items() 
        }
        
        # network_schema_encoder_embedding: [num_nodes_D x emb_dim]
        network_schema_encoder_embedding = self.network_schema_encoder(
            relation_subgraph_dict = self.relation_subgraph_dict,
            relation_neighbor_size_dict = self.relation_neighbor_size_dict,
            feat_dict = h_dict, 
        )
        
        # metapath_encoder_embedding: [num_nodes_D x emb_dim]
        metapath_encoder_embedding = self.metapath_encoder(
            metapath_subgraph_list = self.metapath_subgraph_list,
            feat = h_dict[self.infer_node_type],
        )
        
        contrastive_loss = self.contrast_model(
            network_schema_encoder_embedding = network_schema_encoder_embedding,
            metapath_encoder_embedding = metapath_encoder_embedding,
            positive_sample_mask = self.positive_sample_mask,
        )
        
        return contrastive_loss 
    
    def calc_node_embedding(self) -> FloatTensor:
        with torch.no_grad():
            feat = self.feat_dict[self.infer_node_type]
            
            h = F.elu(
                self.in_fc_dict[self.infer_node_type](feat)
            )
            
            metapath_encoder_embedding = self.metapath_encoder(
                metapath_subgraph_list = self.metapath_subgraph_list,
                feat = h,
            ) 

        return metapath_encoder_embedding