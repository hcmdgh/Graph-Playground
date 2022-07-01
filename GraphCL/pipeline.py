from util import * 


def GraphCL_pipeline(
    graph: dgl.DGLGraph, 
    seed: Optional[int] = None, 
):
    init_log()
    device = auto_set_device()
    
    if seed:
        seed_all(seed)
        
    num_nodes = graph.num_nodes() 
    feat = graph.ndata['feat']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    label = graph.ndata['label']
    
    