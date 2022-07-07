from util import * 
from model import * 
from config import * 

from dl import * 


@dataclass
class Pipeline:
    graph: dgl.DGLGraph 
    hidden_dim: int = 512
    num_epochs: int = 300 
    epsilon: float = 0.01 
    lr: float = 0.001 
    weight_decay: float = 0.
    use_wandb: bool = True 
    
    def run(self):
        set_cwd(__file__)
        init_log()
        device = auto_set_device()
        
        if self.use_wandb:        
            wandb.init(
                project = 'MVGRL',
                config = asdict(self), 
            )
        
        graph = self.graph.to(device)
        num_nodes = graph.num_nodes() 
        feat = graph.ndata.pop('feat')
        feat_dim = feat.shape[-1]
        label = graph.ndata.pop('label')
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')
        
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        
        print("Computing PPR...", flush=True)
        PPR_mat = calc_PPR_mat(graph)
        print("Computing end!", flush=True)
        
        diff_edge_index = np.nonzero(PPR_mat)
        diff_edge_weight = PPR_mat[diff_edge_index]
        diff_edge_weight = torch.tensor(diff_edge_weight, dtype=torch.float32, device=device)

        diff_graph = dgl.graph(diff_edge_index)
        diff_graph = dgl.remove_self_loop(diff_graph)
        diff_graph = dgl.add_self_loop(diff_graph)
        diff_graph = diff_graph.to(device)
        print(diff_graph)
        
        if not USE_DIFF_GRAPH:
            diff_graph = graph 
            diff_edge_weight = None 

        model = MVGRL(
            in_dim = feat_dim,
            out_dim = self.hidden_dim,
        )
        
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_f1_micro = 0. 
        
        for epoch in range(1, self.num_epochs + 1):
            loss = model.train_graph(
                graph = graph,
                diff_graph = diff_graph,
                feat = feat,
                edge_weight = diff_edge_weight,
            )
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            
            log_multi(
                wandb_log = self.use_wandb,
                epoch = epoch,
                loss = float(loss),
            )
            
            if epoch % 5 == 0:
                eval_res = model.eval_graph(
                    graph = graph,
                    diff_graph = diff_graph,
                    feat = feat,
                    edge_weight = diff_edge_weight,
                    label = label,
                    train_mask = train_mask,
                    val_mask = val_mask,
                    test_mask = test_mask, 
                )        
                
                val_f1_micro = eval_res['val_f1_micro']
                val_f1_macro = eval_res['val_f1_macro']
                test_f1_micro = eval_res['test_f1_micro']
                test_f1_macro = eval_res['test_f1_macro']

                log_multi(
                    wandb_log = self.use_wandb,
                    epoch = epoch,
                    val_f1_micro = val_f1_micro,
                    val_f1_macro = val_f1_macro,
                    test_f1_micro = test_f1_micro,
                    test_f1_macro = test_f1_macro,
                )
                
                if self.use_wandb and val_f1_micro > best_val_f1_micro:
                    best_val_f1_micro = val_f1_micro
                    
                    wandb.summary['best_epoch'] = epoch 
                    wandb.summary['best_val_f1_micro'] = val_f1_micro
                    wandb.summary['val_f1_macro'] = val_f1_macro
                    wandb.summary['test_f1_micro'] = test_f1_micro
                    wandb.summary['test_f1_macro'] = test_f1_macro
        
        
if __name__ == '__main__':
    pipeline = Pipeline(
        graph = load_dgl_dataset('cora'), 
    )
    
    pipeline.run() 
