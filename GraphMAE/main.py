from model import * 
from dl import * 


@dataclass
class GraphMAE_pipeline:
    graph: dgl.DGLGraph
    seed: Optional[int] = None 
    raw_feat_classification: bool = False
    gat_hidden_dim: int = 64
    gat_dropout: float = 0.1 
    emb_dim: int = 64
    num_epochs: int = 400 
    lr: float = 0.001
    weight_decay: float = 2e-4

    def run(self):
        init_log()
        device = auto_set_device()
        seed_all(self.seed)

        wandb.init(
            project = 'GraphMAE', 
            config = asdict(self), 
        )
        
        graph = self.graph.to(device)
        
        feat = graph.ndata['feat']
        feat_dim = feat.shape[-1]
        label = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']

        if self.raw_feat_classification:
            print("直接对原始特征进行分类：")

            clf_res = sklearn_multiclass_classification(
                feat = feat,
                label = label,
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
                hidden_dim = self.gat_hidden_dim,
                feat_dropout = self.gat_dropout,
                attn_dropout = self.gat_dropout,
            ),
            decoder_gat_param = GAT.Param(
                in_dim = self.emb_dim,
                out_dim = feat_dim, 
                hidden_dim = self.gat_hidden_dim,
                feat_dropout = self.gat_dropout,
                attn_dropout = self.gat_dropout,
            ),
        )
        
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        best_val_f1_micro = 0. 
        
        for epoch in tqdm(range(1, self.num_epochs + 1), disable=True):
            loss = model.train_graph(g=graph, feat=feat)
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            eval_res = model.eval_graph(
                g = graph,
                feat = feat,
                label = label,
                train_mask = train_mask,
                val_mask = val_mask, 
                test_mask = test_mask, 
            )
            
            log_multi(
                wandb_log = True, 
                epoch = epoch,
                loss = float(loss),
                val_f1_micro = eval_res['val_f1_micro'], 
                val_f1_macro = eval_res['val_f1_macro'], 
                test_f1_micro = eval_res['test_f1_micro'], 
                test_f1_macro = eval_res['test_f1_macro'],
            )
            
            if eval_res['val_f1_micro'] > best_val_f1_micro:
                best_val_f1_micro = eval_res['val_f1_micro']
                wandb.summary['best_val_f1_micro'] = best_val_f1_micro
                

if __name__ == '__main__':
    print(os.getcwd())
    exit() 
    
    pipeline = GraphMAE_pipeline(
        graph = load_dgl_dataset('cora'),
        raw_feat_classification = True, 
    )

    pipeline.run() 
