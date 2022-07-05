from .model import * 
from util import * 

__all__ = ['GraphMAE_pipeline']


@dataclass
class GraphMAE_pipeline:
    graph: dgl.DGLGraph
    seed: Optional[int] 
    raw_feat_classification: bool = False
    gat_hidden_dim: int = 8
    emb_dim: int = 64
    lr: float = 0.001
    weight_decay: float = 2e-4

    def run(self):
        init_log()
        device = auto_set_device()

        if self.seed:
            seed_all(self.seed)
        
        graph = self.graph.to(device)
        
        feat = graph.ndata['feat']
        feat_dim = feat.shape[-1]
        label = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']

        if self.raw_feat_classification:
            print("直接对原始特征进行分类：")

            clf_res = mlp_multiclass_classification(
                feat = feat,
                label = label,
                num_layers = 2,
                train_mask = train_mask,
                val_mask = val_mask,
                test_mask = test_mask,     
            )
            
            print(clf_res)
            print()
        
        model = GraphMAE(
            in_dim = feat_dim,
            emb_dim = self.emb_dim,
            encoder_gat_param = GAT.Param(
                in_dim = feat_dim,
                out_dim = self.emb_dim, 
            ),
            decoder_gat_param = GAT.Param(
                in_dim = self.emb_dim,
                out_dim = feat_dim, 
            ),
        )
        
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        for epoch in itertools.count(1):
            loss = model.train_graph(g=graph, feat=feat)
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            log_multi(
                epoch = epoch,
                loss = float(loss), 
            )
            
            if epoch % 5 == 0:
                eval_res = model.eval_graph(
                    g = graph,
                    feat = feat,
                    label = label,
                    train_mask = train_mask,
                    val_mask = val_mask, 
                    test_mask = test_mask, 
                )
                
                print(eval_res)
