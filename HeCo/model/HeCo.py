from graph import * 
from .MetapathEncoder import * 
from .ContrastiveLoss import * 
from .SchemaEncoder import * 

from dl import * 

__all__ = ['HeCo']


class HeCo(nn.Module):
    def __init__(self, 
                 graph: HeCoGraph,  
                 emb_dim: int, 
                 feat_dropout: float, 
                 attn_dropout: float, 
                 tau: float, 
                 lam: float):
        super().__init__()

        self.graph = graph 
        
        self.in_fc_dict = nn.ModuleDict({
            node_type: nn.Linear(graph.feat_dict[node_type].shape[-1], emb_dim, bias=True)
            for node_type in graph.hg.ntypes 
        })

        self.feat_dropout = nn.Dropout(feat_dropout)

        self.metapath_encoder = MetapathEncoder(
            graph = graph, 
            emb_dim = emb_dim,
            attn_dropout = attn_dropout, 
        )
        
        self.schema_encoder = SchemaEncoder(
            graph = graph, 
            emb_dim = emb_dim,
            attn_dropout = attn_dropout, 
        )   
        
        self.contrastive_loss = ContrastiveLoss(
            emb_dim = emb_dim, 
            tau = tau, 
            lam = lam,
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for fc in self.in_fc_dict.values():
            nn.init.xavier_normal_(fc.weight, gain=1.414)
            nn.init.zeros_(fc.bias)

    def forward(self):
        feat_dict = {
            node_type: F.elu(
                self.feat_dropout(
                    self.in_fc_dict[node_type](self.graph.feat_dict[node_type])
                )
            )
            for node_type in self.graph.hg.ntypes
        }
        
        h_mp = self.metapath_encoder(feat_dict[self.graph.infer_node_type])
        h_sc = self.schema_encoder(feat_dict=feat_dict)
        loss = self.contrastive_loss(h_mp, h_sc, self.graph.positive_sample)
        return loss

    def calc_emb(self) -> FloatTensor:
        self.eval() 
        
        with torch.no_grad():
            infer_node_type = self.graph.infer_node_type 
            h = F.elu(self.in_fc_dict[infer_node_type](self.graph.feat_dict[infer_node_type]))
            h = self.metapath_encoder(h)
        
        return h.detach()

    def eval_graph(self) -> dict[str, float]:
        self.eval() 
        
        infer_ntype = self.graph.infer_node_type 
        
        label = self.graph.hg.nodes[infer_ntype].data['label']
        train_mask = self.graph.hg.nodes[infer_ntype].data['train_mask_20']
        val_mask = self.graph.hg.nodes[infer_ntype].data['val_mask_20']
        test_mask = self.graph.hg.nodes[infer_ntype].data['test_mask_20']
        
        emb = self.calc_emb()
        
        clf_res = sklearn_multiclass_classification(
            feat = emb,
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask, 
            max_epochs = 500,
        )
        
        return clf_res 
