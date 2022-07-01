from util import * 
from .dataset import * 
from .util import * 
from . import config
from .model import * 



def main():
    init_log()
    device = auto_set_device()
    
    if config.dataset == 'cora':
        g = load_Cora_dataset()
        g = g.to(device)
    else:
        raise AssertionError

    feat = g.ndata['feat']
    feat_dim = feat.shape[-1]
    
    if config.mode == 'structure_refinement':
        adj_mat = g.adj().to_dense().to(device)
        anchor_adj_mat = normalize_adj_mat(adj_mat, mode='sym')
    elif config.mode == 'structure_inference':
        raise NotImplementedError 
    else:
        raise AssertionError

    model = SUBLIME(
        graph_encoder_param = GraphEncoder.Param(
            in_dim = feat_dim,
        ),
        feat = feat, 
    )
       
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay) 
    
    for epoch in range(1, 2000):
        model.train() 
        
        loss, learned_adj_mat = model(
            feat = feat,
            anchor_adj_mat = anchor_adj_mat,
        )
    
        
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad() 
        
        log_dict(
            epoch = epoch,
            loss = float(loss), 
        )


if __name__ == '__main__':
    main() 
