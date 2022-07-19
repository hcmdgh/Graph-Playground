from util import * 
from model import * 

from dl import * 


@dataclass
class Config:
    graph: dgl.DGLGraph 
    emb_dim: int = 512
    num_epochs: int = 300 
    epsilon: float = 0.01 
    lr: float = 0.001 
    weight_decay: float = 0.
    
    
def main(config: Config):
    set_cwd(__file__)
    init_log()
    device = auto_set_device()
    
    graph = config.graph.to(device)
    feat = graph.ndata.pop('feat')
    feat_dim = feat.shape[-1]
    label = graph.ndata.pop('label')
    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')
    
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    
    print("Computing PPR...", flush=True)
    PPR_mat = calc_PPR_mat(graph)
    print("Computing end!", flush=True)
    
    diff_edge_index = np.nonzero(PPR_mat)
    diff_edge_weight = PPR_mat[diff_edge_index]
    diff_edge_weight = torch.tensor(diff_edge_weight, dtype=torch.float32, device=device)

    diff_graph = dgl.graph(diff_edge_index)
    diff_graph = dgl.remove_self_loop(diff_graph)
    diff_graph = dgl.add_self_loop(diff_graph)
    diff_graph = diff_graph.to(device)
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
                    
        h_g_f = model.gnn_encoder_1(graph, feat)
        h_d_f = model.gnn_encoder_2(diff_graph, feat, edge_weight=diff_edge_weight)
        
        perm = np.random.permutation(len(feat))
        shuffled_feat = feat[perm]
        
        h_g_s = model.gnn_encoder_1(graph, shuffled_feat)
        h_d_s = model.gnn_encoder_2(diff_graph, shuffled_feat, edge_weight=diff_edge_weight)
        
        p_g_f = model.act(model.pooling(graph, h_g_f))
        p_d_f = model.act(model.pooling(diff_graph, h_d_f))
        
        logits = model.discriminator(
            h_g_f = h_g_f,
            h_d_f = h_d_f,
            h_g_s = h_g_s,
            h_d_s = h_d_s,
            p_g_f = p_g_f,
            p_d_f = p_d_f,
        )
        
        num_nodes = len(feat)
        
        target = torch.cat(
            [
                torch.ones(num_nodes * 2),
                torch.zeros(num_nodes * 2),
            ],
            dim = 0,
        ).to(device)
        
        loss = F.binary_cross_entropy_with_logits(input=logits, target=target)
        
        return loss  
    
    
    def eval_epoch(mask: BoolArrayTensor) -> float:
        model.eval() 
        
        with torch.no_grad():
            h_g_f = model.gnn_encoder_1(graph, feat)
            h_d_f = model.gnn_encoder_2(diff_graph, feat, edge_weight=diff_edge_weight)
            
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
    main(
        Config(
            graph = load_dgl_dataset('cora'), 
        )
    )
