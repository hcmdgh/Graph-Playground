from .Discriminator import * 
from config import * 

from dl import * 

__all__ = ['MVGRL']


class MVGRL(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 act: Callable = torch.sigmoid):
        super().__init__()
        
        self.gnn_encoder_1 = dglnn.GraphConv(
            in_feats = in_dim,
            out_feats = out_dim,
            norm = 'both',
            activation = nn.PReLU(), 
        )
        
        self.gnn_encoder_2 = dglnn.GraphConv(
            in_feats = in_dim,
            out_feats = out_dim,
            norm = 'none',
            activation = nn.PReLU(), 
        )
        
        self.pooling = dglnn.AvgPooling()
        
        self.discriminator = Discriminator(emb_dim=out_dim)
        
        self.act = act  
        
        self.device = get_device()
        self.to(self.device)
        
    def calc_emb(self,
                 graph: dgl.DGLGraph,
                 diff_graph: dgl.DGLGraph,
                 feat: FloatTensor,
                 edge_weight: FloatTensor) -> FloatTensor:
        self.eval()
                 
        with torch.no_grad():
            h_g_f = self.gnn_encoder_1(graph, feat)
            h_d_f = self.gnn_encoder_2(diff_graph, feat, edge_weight=edge_weight)
            
        out = (h_g_f + h_d_f).detach() 
        
        return out 
        
    def train_graph(self,
                    graph: dgl.DGLGraph,
                    diff_graph: dgl.DGLGraph,
                    feat: FloatTensor,
                    edge_weight: FloatTensor) -> FloatScalarTensor:
        self.train() 
                    
        h_g_f = self.gnn_encoder_1(graph, feat)
        h_d_f = self.gnn_encoder_2(diff_graph, feat, edge_weight=edge_weight)
        
        if USE_NODE_FEAT_SHUFFLE:
            perm = np.random.permutation(len(feat))
            shuffled_feat = feat[perm]
        else:
            shuffled_feat = feat 
        
        h_g_s = self.gnn_encoder_1(graph, shuffled_feat)
        h_d_s = self.gnn_encoder_2(diff_graph, shuffled_feat, edge_weight=edge_weight)
        
        p_g_f = self.act(self.pooling(graph, h_g_f))
        p_d_f = self.act(self.pooling(diff_graph, h_d_f))
        
        logits = self.discriminator(
            h_g_f = h_g_f,
            h_d_f = h_d_f,
            h_g_s = h_g_s,
            h_d_s = h_d_s,
            p_g_f = p_g_f,
            p_d_f = p_d_f,
        )
        
        num_nodes = len(feat)
        
        target = torch.cat(
            [
                torch.ones(num_nodes * 2),
                torch.zeros(num_nodes * 2),
            ],
            dim = 0,
        ).to(self.device)
        
        loss = F.binary_cross_entropy_with_logits(input=logits, target=target)
        
        return loss  

    def eval_graph(self,
                   graph: dgl.DGLGraph,
                   diff_graph: dgl.DGLGraph,
                   feat: FloatTensor,
                   edge_weight: FloatTensor,
                   label: IntTensor, 
                   train_mask: BoolTensor,
                   val_mask: BoolTensor,
                   test_mask: BoolTensor) -> dict[str, float]:
        self.eval() 
        
        emb = self.calc_emb(
            graph = graph,
            diff_graph = diff_graph,
            feat = feat,
            edge_weight = edge_weight, 
        )
        
        clf_res = sklearn_multiclass_classification(
            feat = emb,
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
            max_epochs = 300,    
        )
        
        return clf_res 
