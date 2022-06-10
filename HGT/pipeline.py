from util import * 
from .model import * 


def HGT_pipeline(*,
                 hg_path: str,
                 use_gpu: bool = True,
                 official: bool, 
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 num_heads: int = 2,
                 lr: float = 0.005,
                 weight_decay: float = 0.001,
                 early_stopping_epochs: int = 50,):
    init_log('./log.log')
                 
    auto_set_device(use_gpu=use_gpu)
    device = get_device() 
                 
    hg_graph = HeteroGraph.load_from_file(hg_path) 
    
    hg = hg_graph.to_dgl().to(device)

    assert len(hg_graph.node_attr_dict['label']) == 1
    infer_node_type = next(iter(hg_graph.node_attr_dict['label']))
    
    feat_dict = to_device(hg_graph.node_attr_dict['feat'])
    label_th = hg_graph.node_attr_dict['label'][infer_node_type].to(device)
    label = label_th.cpu().numpy() 
    train_mask = hg_graph.node_attr_dict['train_mask'][infer_node_type].numpy() 
    val_mask = hg_graph.node_attr_dict['val_mask'][infer_node_type].numpy() 
    test_mask = hg_graph.node_attr_dict['test_mask'][infer_node_type].numpy() 
    
    if official:
        HGT_Class = HGT_Official
    else:
        HGT_Class = HGT 
    
    model = HGT_Class(
        in_dim = {
            node_type: feat.shape[-1]
            for node_type, feat in hg_graph.node_attr_dict['feat'].items()
        },
        hidden_dim = hidden_dim,
        out_dim = hg_graph.num_classes, 
        node_types = hg_graph.node_types,
        edge_types = hg_graph.edge_types,
        num_heads = num_heads,
        num_layers = num_layers, 
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    early_stopping = EarlyStopping(
        monitor_field = 'val_acc',
        tolerance_epochs = early_stopping_epochs,
        expected_trend = 'asc',
    )
    
    for epoch in itertools.count(1):
        model.train() 
        
        logits_dict = model(hg=hg, feat_dict=feat_dict)
        logits = logits_dict[infer_node_type]
        
        loss = F.cross_entropy(input=logits[train_mask], target=label_th[train_mask])
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
        
        model.eval()   
        
        with torch.no_grad():
            logits_dict = model(hg=hg, feat_dict=feat_dict)
            logits = logits_dict[infer_node_type]      

        y_pred = logits[val_mask].detach().cpu().numpy() 
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = label[val_mask]
        val_acc = calc_acc(y_pred=y_pred, y_true=y_true)
        
        y_pred = logits[test_mask].detach().cpu().numpy() 
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = label[test_mask]
        test_acc = calc_acc(y_pred=y_pred, y_true=y_true)
        
        early_stopping.record(
            epoch = epoch, 
            val_acc = val_acc,
            test_acc = test_acc, 
        )

        logging.info(f"epoch: {epoch}, train_loss: {float(loss):.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")

        early_stopping.auto_stop()
