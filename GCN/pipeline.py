from util import * 
from .model import * 


def GCN_pipeline(*,
                 homo_graph_path: str,
                 use_gpu: bool = True,
                 add_self_loop: bool = True,
                 hidden_dim: int = 16,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 lr: float = 0.01,
                 weight_decay: float = 5e-4,
                 early_stopping_epochs: int = 50):
    init_log('./log.log')
    auto_set_device(use_gpu=use_gpu)
    device = get_device()
    
    homo_graph = HomoGraph.load_from_file(homo_graph_path)
    g = homo_graph.to_dgl()
    
    feat = homo_graph.node_attr_dict['feat'].to(device)
    feat_dim = feat.shape[-1]
    label = homo_graph.node_attr_dict['label'].to(device)
    label_np = label.cpu().numpy()
    train_mask = homo_graph.node_attr_dict['train_mask'].numpy()
    val_mask = homo_graph.node_attr_dict['val_mask'].numpy()
    test_mask = homo_graph.node_attr_dict['test_mask'].numpy()
    
    if add_self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        
    g = g.to(device)
    
    model = GCN(
        in_dim = feat_dim,
        hidden_dim = hidden_dim,
        out_dim = homo_graph.num_classes,
        num_layers = num_layers,
        activation = torch.relu,
        dropout = dropout,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    early_stopping = EarlyStopping(
        monitor_field = 'val_acc',
        tolerance_epochs = early_stopping_epochs,
        expected_trend = 'asc', 
    )
    
    for epoch in itertools.count(1):
        model.train() 
        
        logits = model(g, feat)
        
        loss = F.cross_entropy(input=logits[train_mask], target=label[train_mask])
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        
        
        model.eval() 
        
        with torch.no_grad():
            logits = model(g, feat)
            
        y_pred = logits.detach().cpu().numpy() 
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = label_np 
        
        val_acc = calc_acc(y_pred=y_pred[val_mask], y_true=y_true[val_mask])
        test_acc = calc_acc(y_pred=y_pred[test_mask], y_true=y_true[test_mask])

        early_stopping.record(
            epoch = epoch,
            val_acc = val_acc,
            test_acc = test_acc,
        )
        
        logging.info(f"epoch: {epoch}, loss: {float(loss):.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")
        
        early_stopping.auto_stop()
