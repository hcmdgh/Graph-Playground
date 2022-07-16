from model import * 

from dl import * 


@dataclass
class Config:
    graph: dgl.DGLGraph
    add_self_loop: bool = True 
    drop_feat_prop_1: float = 0.3
    drop_feat_prop_2: float = 0.4
    drop_edge_prop_1: float = 0.2
    drop_edge_prop_2: float = 0.4
    num_gnn_layers: int = 2
    emb_dim: int = 128
    tau: float = 0.4
    train_val_ratio: tuple[float, float] = (0.9, 0.1)
    num_epochs: int = 200
    lr: float = 0.001
    weight_decay: float = 1e-5


def main(config: Config):
    set_cwd(__file__)
    init_log()
    device = auto_set_device()
    
    graph = config.graph.to(device)
    
    feat = graph.ndata.pop('feat') 
    feat_dim = feat.shape[-1]

    label = graph.ndata.pop('label') 
    label_np = label.cpu().numpy() 
    
    train_mask, val_mask, _ = split_train_val_test_set(
        total_cnt = graph.num_nodes(),
        train_ratio = config.train_val_ratio[0],
        val_ratio = config.train_val_ratio[1],
    )
    test_mask = val_mask
    
    # train_mask = graph.ndata.pop('train_mask')
    # val_mask = graph.ndata.pop('val_mask')
    # test_mask = graph.ndata.pop('test_mask')
    
    if config.add_self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
    
    model = GRACE(
        in_dim = feat_dim,
        emb_dim = config.emb_dim,
        num_gnn_layers = config.num_gnn_layers,
        tau = config.tau,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    metric_recorder = MetricRecorder() 
    
    wandb.init(project='GRACE', config=asdict(config))

    for epoch in itertools.count(1):
        model.train() 
        
        dropped_g_1 = drop_edge(graph, drop_prob=config.drop_edge_prop_1)
        dropped_g_2 = drop_edge(graph, drop_prob=config.drop_edge_prop_2)
        dropped_feat_1 = drop_feature(feat, drop_prob=config.drop_feat_prop_1)
        dropped_feat_2 = drop_feature(feat, drop_prob=config.drop_feat_prop_2)

        # [BEGIN] 增加自环
        dropped_g_1 = dgl.remove_self_loop(dropped_g_1)
        dropped_g_1 = dgl.add_self_loop(dropped_g_1)
        dropped_g_2 = dgl.remove_self_loop(dropped_g_2)
        dropped_g_2 = dgl.add_self_loop(dropped_g_2)
        # [END]
        
        z1 = model(g=dropped_g_1, feat=dropped_feat_1)
        z2 = model(g=dropped_g_2, feat=dropped_feat_2)
        
        loss = model.calc_contrastive_loss(z1=z1, z2=z2)
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
        logging.info(f"epoch: {epoch}, loss: {float(loss):.4f}")

        if epoch % 5:
            metric_recorder.record(
                epoch = epoch,
                log = True,
                wandb_log = True,
                loss = loss, 
            )
        else:
            model.eval() 
            
            with torch.no_grad():
                emb = model(g=graph, feat=feat)
            
            clf_res = sklearn_multiclass_classification(
                feat = emb.detach().cpu().numpy(),
                label = label_np,
                train_mask = train_mask,
                val_mask = val_mask,
                test_mask = test_mask,
                max_epochs = 300,
            )
            
            metric_recorder.record(
                epoch = epoch,
                log = True,
                wandb_log = True,
                loss = loss, 
                val_f1_micro = clf_res['val_f1_micro'],
                val_f1_macro = clf_res['val_f1_macro'],
                test_f1_micro = clf_res['test_f1_micro'],
                test_f1_macro = clf_res['test_f1_macro'],
            )
            
            if metric_recorder.check_early_stopping('val_f1_micro', expected_trend='asc', tolerance=50):
                break 
            
    metric_recorder.best_record(
        field_name = 'val_f1_micro',
        log = True,
        wandb_log = True, 
    )


if __name__ == '__main__':
    main(
        Config(
            graph = load_dgl_dataset('cora'), 
        )
    )            
