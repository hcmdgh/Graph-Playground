from util import * 
from model import * 
from config import * 

from dl import * 

    
def main():
    set_cwd(__file__)
    init_log()
    device = auto_set_device(use_gpu=False)
    
    graph = config.graph.to(device)
    num_nodes = graph.num_nodes() 
    feat = graph.ndata.pop('feat')
    feat_dim = feat.shape[-1]
    label = graph.ndata.pop('label')
    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')
    
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    
    print("graph:")
    print(graph)
    
    if config.graph_diffusion == 'PPR':
        diff_graph, diff_edge_weight = generate_PPR_graph(graph)
    elif config.graph_diffusion == 'APPNP': 
        diff_graph, diff_edge_weight = generate_APPNP_graph(graph)
    else:
        raise AssertionError 

    diff_graph = diff_graph.to(device)
    diff_edge_weight = diff_edge_weight.to(device)
    
    print("diff_graph:")
    print(diff_graph)

    model = MVGRL(
        in_dim = feat_dim,
        out_dim = config.emb_dim,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    recorder = ClassificationRecorder(model=model)
    
    wandb.init(project='MVGRL', config=asdict(config))
    
    
    def train_epoch() -> FloatScalarTensor:
        model.train() 
                    
        l_g_f = model.gnn_encoder_1(graph, feat)
        l_dg_f = model.gnn_encoder_2(diff_graph, feat, edge_weight=diff_edge_weight)
        
        perm = np.random.permutation(len(feat))
        shuffled_feat = feat[perm]
        
        l_g_sf = model.gnn_encoder_1(graph, shuffled_feat)
        l_dg_sf = model.gnn_encoder_2(diff_graph, shuffled_feat, edge_weight=diff_edge_weight)
        
        g_g_f = model.act(model.pooling(graph, l_g_f))
        g_dg_f = model.act(model.pooling(diff_graph, l_dg_f))
        
        loss = model.discriminator(
            l_g_f = l_g_f,
            l_dg_f = l_dg_f,
            l_g_sf = l_g_sf,
            l_dg_sf = l_dg_sf,
            g_g_f = g_g_f,
            g_dg_f = g_dg_f,
        )
        
        return loss  
    
    
    def eval_epoch(mask: BoolArrayTensor) -> float:
        model.eval() 
        
        with torch.no_grad():
            h_g_f = model.gnn_encoder_1(graph, feat)
            h_d_f = model.gnn_encoder_2(diff_graph, feat, edge_weight=diff_edge_weight)
            
        if config.use_encoder_1_as_emb:
            emb = h_g_f.detach()
        else:
            emb = (h_g_f + h_d_f).detach() 
        
        eval_acc = mlp_multiclass_classification(
            feat = emb,
            label = label,
            train_mask = train_mask,
            val_mask = mask,
            num_epochs = 300,    
        )['val_f1_micro']
        
        return eval_acc
    
    
    for epoch in range(1, config.num_epochs + 1):
        loss = train_epoch()
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
        recorder.train(epoch=epoch, loss=loss)
        
        if epoch % 5 == 0:
            val_acc = eval_epoch(mask=val_mask)  
             
            recorder.validate(epoch=epoch, val_acc=val_acc)
            
    recorder.load_best_model_state() 
    
    test_acc = eval_epoch(mask=test_mask)
            
    recorder.test(test_acc=test_acc)
        
        
if __name__ == '__main__':
    main()
