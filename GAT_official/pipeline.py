from .model import *
from util import * 
import wandb 

__all__ = ['GAT_pipeline', 'GAT']


def GAT_pipeline(
    graph: dgl.DGLGraph,
    GAT_param: GAT.Param,
    seed: Optional[int] = None, 
    add_self_loop: bool = True, 
    lr: float = 0.005,
    weight_decay: float = 0.0005,
    num_epochs: int = 200,
):
    init_log()
    device = auto_set_device()

    seed_all(seed)
    
    wandb.init(project='GAT_official')
    wandb.config.dropout = GAT_param.feat_dropout  
    
    print(graph)
    
    graph = graph.to(device)
    feat = graph.ndata['feat']
    label = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    feat_dim = feat.shape[-1]
    num_classes = int(torch.max(label)) + 1 
    
    GAT_param.in_dim = feat_dim
    GAT_param.out_dim = num_classes

    if add_self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

    model = GAT(GAT_param)
    
    wandb.watch(model, log_freq=20)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_f1_micro = 0. 

    for epoch in tqdm(range(1, num_epochs + 1)):
        # Train        
        loss = model.train_graph(g=graph, feat=feat, mask=train_mask, label=label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Val & Test
        val_f1_micro, val_f1_macro = model.eval_graph(g=graph, feat=feat, mask=val_mask, label=label)
        test_f1_micro, test_f1_macro = model.eval_graph(g=graph, feat=feat, mask=test_mask, label=label)

        if val_f1_micro > best_val_f1_micro:
            best_val_f1_micro = val_f1_micro 
            
            wandb.run.summary['best_val_f1_micro'] = best_val_f1_micro
        
        # log_multi(
        #     epoch = epoch, 
        #     loss = float(loss), 
        #     val_f1_micro = val_f1_micro,
        #     val_f1_macro = val_f1_macro,
        #     test_f1_micro = test_f1_micro,
        #     test_f1_macro = test_f1_macro,
        # )        
        
        wandb.log(dict(
            epoch = epoch, 
            loss = float(loss), 
            val_f1_micro = val_f1_micro,
            val_f1_macro = val_f1_macro,
            test_f1_micro = test_f1_micro,
            test_f1_macro = test_f1_macro,
        ))
