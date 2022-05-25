from util import *
from dataset.acm import * 
from dataset.ogbn_mag import * 
from .config import * 
from .model.han import * 


def main():
    set_device(DEVICE)
    
    init_log('./log.log')
    
    if DATASET == 'acm':
        graph_dataset = load_acm_dataset()
    elif DATASET == 'ogbn_mag':
        graph_dataset = load_ogbn_mag_dataset() 
    else:
        raise AssertionError
        
    hg = graph_dataset.to_dgl() 
    hg = to_device(hg)
    
    feat = to_device(graph_dataset.node_prop_dict['feat']['paper'])
    
    train_mask = graph_dataset.node_prop_dict['train_mask']['paper'].cpu().numpy()
    val_mask = graph_dataset.node_prop_dict['val_mask']['paper'].cpu().numpy()
    test_mask = graph_dataset.node_prop_dict['test_mask']['paper'].cpu().numpy()
    assert np.all(train_mask | val_mask | test_mask)
    assert np.all(~(train_mask & val_mask & test_mask))
    
    label = to_device(graph_dataset.node_prop_dict['label']['paper'])
    label_np = label.cpu().numpy() 
    
    model = HAN(
        hg = hg,
        metapaths = METAPATHS,
        in_dim = feat.shape[-1],
        hidden_dim = HIDDEN_DIM,
        out_dim = graph_dataset.num_classes,
        num_layers = NUM_LAYERS,
        layer_num_heads = LAYER_NUM_HEADS,
        dropout_ratio = DROPOUT_RATIO,
    )
    model = to_device(model)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    val_metric = MultiClassificationMetric(status='val')
    test_metric = MultiClassificationMetric(status='test')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        def train_epoch():
            model.train()
            
            logits = model(feat)
            
            loss = F.cross_entropy(input=logits[train_mask], target=label[train_mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            logging.info(f"epoch: {epoch}, train_loss: {float(loss):.4f}")  
            
        def val():
            model.eval()
            
            with torch.no_grad(): 
                logits = model(feat) 
                
            y_true = label_np[val_mask]
            
            y_pred = np.argmax(logits.cpu().numpy()[val_mask], axis=-1) 
            
            val_metric.measure(epoch=epoch, y_true=y_true, y_pred=y_pred)  
            
        def test():
            model.eval()
            
            with torch.no_grad(): 
                logits = model(feat) 
                
            y_true = label_np[test_mask]
            
            y_pred = np.argmax(logits.cpu().numpy()[test_mask], axis=-1) 
            
            test_metric.measure(epoch=epoch, y_true=y_true, y_pred=y_pred)  
            
        train_epoch()  
        
        if epoch % 10 == 0:
            val()
            test() 
    
    
if __name__ == '__main__':
    main() 
