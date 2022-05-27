from util import * 
from dataset.ACDNE_dataset import * 
from .config import * 
from .model.gat import * 
from .util import * 


def main():
    set_device(DEVICE)
    init_log('./log.log')
    
    src_dataset = load_ACDNE_dataset(SRC_DATASET)
    tgt_dataset = load_ACDNE_dataset(TGT_DATASET)
    src_g = src_dataset.to_dgl()
    tgt_g = tgt_dataset.to_dgl()
    
    feat_dim = src_dataset.node_prop_dict['feat'].shape[-1]
    # src_feat = to_device(src_dataset.node_prop_dict['feat'])
    # tgt_feat = to_device(tgt_dataset.node_prop_dict['feat'])
    # assert src_feat.shape[-1] == tgt_feat.shape[-1]
    # assert src_dataset.num_classes == tgt_dataset.num_classes
    # src_label = to_device(src_dataset.node_prop_dict['label'].to(torch.float32))
    # src_label_np = src_dataset.node_prop_dict['label'].cpu().numpy()
    # tgt_label = to_device(tgt_dataset.node_prop_dict['label'].to(torch.float32))
    # tgt_label_np = tgt_dataset.node_prop_dict['label'].cpu().numpy()
    
    # [BEGIN] 合并源图和目标图    
    src_g.ndata['feat'] = src_dataset.node_prop_dict['feat']
    src_g.ndata['label'] = src_dataset.node_prop_dict['label'].to(torch.float32)
    tgt_g.ndata['feat'] = tgt_dataset.node_prop_dict['feat'] 
    tgt_g.ndata['label'] = tgt_dataset.node_prop_dict['label'].to(torch.float32)
    
    combined_g = combine_graphs(src_g=src_g, tgt_g=tgt_g, add_super_node=True)
    combined_g = to_device(combined_g)
    
    combined_feat = combined_g.ndata['feat']
    combined_label = combined_g.ndata['label']
    combined_label_np = combined_g.ndata['label'].cpu().numpy() 
    
    train_mask = np.zeros(combined_g.num_nodes(), dtype=bool)
    val_mask = np.zeros(combined_g.num_nodes(), dtype=bool)
    origin = combined_g.ndata['origin'].cpu().numpy() 
    train_mask[origin == 1] = True 
    val_mask[origin == 2] = True 
    assert np.sum(train_mask) + np.sum(val_mask) == combined_g.num_nodes() - 2
    # [END]
    
    if MODEL == 'MLP':
        model = MLP(
            in_dim = feat_dim, 
            out_dim = src_dataset.num_classes, 
            num_layers = 2,
        )
        model = to_device(model)
    elif MODEL == 'GAT':
        model = GAT(
            in_dim = feat_dim,
            hidden_dim = 128,
            out_dim = src_dataset.num_classes,
            num_heads = 4,
            num_layers = 2,
            dropout_ratio = 0.0,
            residual = True,
        )
        model = to_device(model)
    else:
        raise AssertionError
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    val_metric = MultiClassificationMetric(status='val')
    
    early_stopping = EarlyStopping()
    
    for epoch in itertools.count(1):
        def train_epoch():
            model.train()
            
            # logits = model(combined_g, combined_feat)
            logits = model(combined_feat)
            
            loss = F.binary_cross_entropy_with_logits(input=logits[train_mask], target=combined_label[train_mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            val_loss = F.binary_cross_entropy_with_logits(input=logits[val_mask], target=combined_label[val_mask])

            early_stopping.record_loss(val_loss)

            logging.info(f"epoch: {epoch}, train_loss: {float(loss):.4f}, val_loss: {float(val_loss):.4f}")  
            
        def val():
            model.eval()
            
            with torch.no_grad(): 
                # logits = model(combined_g, combined_feat) 
                logits = model(combined_feat) 
                
            y_true = combined_label_np[val_mask]
            
            y_pred = np.zeros_like(y_true, dtype=np.int64) 
            logits_np = logits.cpu().numpy()[val_mask]
            y_pred[logits_np > 0] = 1 
            
            val_metric.measure(epoch=epoch, y_true=y_true, y_pred=y_pred)  
            
        train_epoch()  
        
        if epoch % 10 == 0:
            val()

            early_stopping.check_stop() 
    
    
if __name__ == '__main__':
    main() 
