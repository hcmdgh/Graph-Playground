from util import *
from dataset.ogbn_arxiv import * 
from .config import * 
from .model import * 


def main():
    set_device(DEVICE)
    
    init_log('./log.log')
    
    if DATASET == 'ogbn-arxiv':
        graph_dataset = load_ogbn_arxiv_dataset()
    else:
        raise AssertionError
        
    g = graph_dataset.to_dgl() 
    g = to_device(g)
    
    feat = to_device(graph_dataset.node_prop_dict['feat'])
    
    if USE_RANDOM_FEAT:
        feat = to_device(torch.randn_like(feat))
    
    train_mask = graph_dataset.node_prop_dict['train_mask'].cpu().numpy()
    val_mask = graph_dataset.node_prop_dict['val_mask'].cpu().numpy()
    test_mask = graph_dataset.node_prop_dict['test_mask'].cpu().numpy()
    assert np.all(train_mask | val_mask | test_mask)
    assert np.all(~(train_mask & val_mask & test_mask))
    
    label = to_device(graph_dataset.node_prop_dict['label'])
    label_np = label.cpu().numpy() 
    
    model = GAT(
        in_dim = feat.shape[-1],
        hidden_dim = HIDDEN_DIM,
        out_dim = graph_dataset.num_classes,
        num_heads = NUM_HEADS,
        num_layers = NUM_LAYERS,
        dropout_ratio = DROPOUT_RATIO,
        residual = USE_RESIDUAL,
    )
    model = to_device(model)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    val_metric = MultiClassificationMetric(status='val')
    test_metric = MultiClassificationMetric(status='test')
    
    early_stopping = EarlyStopping()
    
    for epoch in itertools.count(1):
        def train_epoch():
            model.train()
            
            logits = model(g, feat)
            
            loss = F.cross_entropy(input=logits[train_mask], target=label[train_mask])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            val_loss = F.cross_entropy(input=logits[val_mask], target=label[val_mask])

            early_stopping.record_loss(val_loss)

            logging.info(f"epoch: {epoch}, train_loss: {float(loss):.4f}, val_loss: {float(val_loss):.4f}")  
            
        def val():
            model.eval()
            
            with torch.no_grad(): 
                logits = model(g, feat) 
                
            y_true = label_np[val_mask]
            
            y_pred = np.argmax(logits.cpu().numpy()[val_mask], axis=-1) 
            
            val_metric.measure(epoch=epoch, y_true=y_true, y_pred=y_pred)  
            
        def test():
            model.eval()
            
            with torch.no_grad(): 
                logits = model(g, feat) 
                
            y_true = label_np[test_mask]
            
            y_pred = np.argmax(logits.cpu().numpy()[test_mask], axis=-1) 
            
            test_metric.measure(epoch=epoch, y_true=y_true, y_pred=y_pred)  
            
        train_epoch()  
        
        if epoch % 10 == 0:
            val()
            test() 

            early_stopping.check_stop() 
    
    
if __name__ == '__main__':
    main() 
