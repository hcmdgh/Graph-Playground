from .model import * 
from util import * 

__all__ = ['GRACE_Pipeline']


def GRACE_Pipeline(
    homo_graph: dgl.DGLGraph,
    drop_feat_prop_1: float = 0.3,
    drop_feat_prop_2: float = 0.4,
    drop_edge_prop_1: float = 0.2,
    drop_edge_prop_2: float = 0.4,
    num_gnn_layers: int = 2,
    emb_dim: int = 128,
    tau: float = 0.4,
    train_val_ratio: tuple[float, float] = (0.9, 0.1),
    num_epochs: int = 200,
    lr: float = 0.0005,
    weight_decay: float = 1e-5,
):
    init_log()
    device = auto_set_device()
    
    g = homo_graph.to(device)
    
    feat = torch.tensor(g.ndata['feat'], dtype=torch.float32, device=device)
    feat_dim = feat.shape[-1]

    label = torch.tensor(g.ndata['label'], dtype=torch.int64, device=device)
    label_np = label.cpu().numpy() 
    
    train_mask, val_mask, _ = split_train_val_test_set(
        total_cnt = g.num_nodes(),
        train_ratio = train_val_ratio[0],
        val_ratio = train_val_ratio[1],
    )
    
    # [BEGIN] 增加自环
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    # [END]
    
    model = GRACE(
        in_dim = feat_dim,
        emb_dim = emb_dim,
        num_gnn_layers = num_gnn_layers,
        tau = tau,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stopping = EarlyStopping(
        monitor_field = 'val_f1_micro',
        tolerance_epochs = 50,
        expected_trend = 'asc', 
    )
    
    for epoch in itertools.count(1):
        model.train() 
        
        dropped_g_1 = drop_edge(g, drop_prob=drop_edge_prop_1)
        dropped_g_2 = drop_edge(g, drop_prob=drop_edge_prop_2)
        dropped_feat_1 = drop_feature(feat, drop_prob=drop_feat_prop_1)
        dropped_feat_2 = drop_feature(feat, drop_prob=drop_feat_prop_2)

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

        if epoch % 1 == 0:
            model.eval() 
            
            with torch.no_grad():
                emb = model(g=g, feat=feat)
            
            xgb_res = xgb_multiclass_classification(
                feat = emb.detach().cpu().numpy(),
                label = label_np,
                train_mask = train_mask,
                val_mask = val_mask,
                check_mask = False,
            )
            val_f1_micro = xgb_res['val_f1_micro']
            val_f1_macro = xgb_res['val_f1_macro']
            
            logging.info(f"epoch: {epoch}, val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}")
            
            early_stopping.record(
                epoch = epoch,
                val_f1_micro = val_f1_micro,
                val_f1_macro = val_f1_macro, 
            )
            
            early_stopping.auto_stop()
