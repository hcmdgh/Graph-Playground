from model import * 

from dl import * 


@dataclass
class Config:
    graph: dgl.DGLGraph
    add_self_loop: bool = True 
    
    emb_dim: int = 600
    num_gcn_layers: int = 2
    dropout: float = 0.1
    
    lr: float = 0.0005
    weight_decay: float = 0.
    
    
def main(config: Config):
    set_cwd(__file__)
    device = auto_set_device()
    init_log()
    
    graph = config.graph.to(device)
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')
    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')
    feat_dim = feat.shape[-1]
    num_classes = int(torch.max(label)) + 1 
    
    # [BEGIN] 原始特征分类
    if True:
        print("原始特征分类：")
        clf_res = mlp_multiclass_classification(
            feat = feat,
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
        )
        print(clf_res)
        print() 
    # [END]

    if config.add_self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

    model = DGI(
        in_dim = feat_dim,
        emb_dim = config.emb_dim,
        num_gcn_layers = config.num_gcn_layers,
        act = nn.PReLU(config.emb_dim),
        dropout = config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    metric_recorder = MetricRecorder()
    
    wandb.init(project='DGI', config=asdict(config))

    for epoch in itertools.count(1):
        model.train()

        optimizer.zero_grad()
        loss = model.train_graph(g=graph, feat=feat)
        loss.backward()
        optimizer.step()

        if epoch % 5:
            metric_recorder.record(
                epoch = epoch,
                log = True,
                wandb_log = True,
                loss = loss, 
            )
        else:
            eval_res = model.eval_graph(
                g = graph,
                feat = feat,
                label = label,
                train_mask = train_mask,
                val_mask = val_mask,
                test_mask = test_mask, 
            )
            
            metric_recorder.record(
                epoch = epoch,
                log = True,
                wandb_log = True,
                loss = loss, 
                val_f1_micro = eval_res['val_f1_micro'],
                val_f1_macro = eval_res['val_f1_macro'],
                test_f1_micro = eval_res['test_f1_micro'],
                test_f1_macro = eval_res['test_f1_macro'],
            )
            
            # pickle_dump(
            #     {
            #         'emb': emb.cpu().numpy(),
            #         'label': label.cpu().numpy(),
            #     },
            #     './output/emb.pkl', 
            # )
            
            if metric_recorder.check_early_stopping('val_f1_micro', 'asc', tolerance=100):
                break 
            
    metric_recorder.best_record(
        'val_f1_micro',
        log = True,
        wandb_log = True, 
    )
                    

if __name__ == '__main__':
    main(
        Config(
            graph = load_dgl_dataset('citeseer'),
            # graph = load_ogb_dataset('ogbn-arxiv'),
        )
    )
