from .model import *
from util import * 

__all__ = ['GCN_pipeline']


def GCN_pipeline(
    graph: dgl.DGLGraph,
    add_self_loop: bool = True, 
    hidden_dim: int = 16,
    num_layers: int = 2, 
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    num_epochs: int = 200,
):
    init_log()
    device = auto_set_device()
    
    graph = graph.to(device)
    feat = graph.ndata['feat']
    label = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    feat_dim = feat.shape[-1]
    num_classes = int(torch.max(label)) + 1 

    if add_self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

    model = GCN(
        in_dim = feat_dim,
        hidden_dim = hidden_dim,
        out_dim = num_classes,
        num_layers = num_layers,
        activation = torch.relu,
        dropout = 0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, num_epochs + 1):
        model.train()

        logits = model(g=graph, feat=feat)
        
        loss = F.cross_entropy(input=logits[train_mask], target=label[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        model.eval() 
        
        with torch.no_grad():
            logits = model(g=graph, feat=feat)
            
        y_pred = torch.argmax(logits[val_mask], dim=-1).detach().cpu().numpy() 
        y_true = label[val_mask].cpu().numpy() 
        
        val_f1_micro = calc_f1_micro(y_true=y_true, y_pred=y_pred)
        val_f1_macro = calc_f1_macro(y_true=y_true, y_pred=y_pred)
        
        log_multi(
            epoch = epoch, 
            loss = float(loss), 
            val_f1_micro = val_f1_micro,
            val_f1_macro = val_f1_macro,
        )        


    model.eval() 
        
    with torch.no_grad():
        logits = model(g=graph, feat=feat)
        
    y_pred = torch.argmax(logits[test_mask], dim=-1).detach().cpu().numpy() 
    y_true = label[test_mask].cpu().numpy() 
    
    test_f1_micro = calc_f1_micro(y_true=y_true, y_pred=y_pred)
    test_f1_macro = calc_f1_macro(y_true=y_true, y_pred=y_pred)
    
    log_multi(
        epoch = 'Final Test', 
        test_f1_micro = test_f1_micro,
        test_f1_macro = test_f1_macro,
    )   
