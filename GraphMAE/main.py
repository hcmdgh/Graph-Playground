from model import * 
from config import * 
from util import * 

from dl import * 


def main():
    set_cwd(__file__)
    init_log()
    device = auto_set_device()

    wandb.init(project='GraphMAE', config=asdict(config))
    
    graph = config.graph.to(device)
    feat = graph.ndata.pop('feat') 
    feat_dim = feat.shape[-1]
    label = graph.ndata.pop('label')
    train_mask = graph.ndata.pop('train_mask') 
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    if False:
        print("直接对原始特征进行分类：")

        clf_res = mlp_multiclass_classification(
            feat = feat,
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,     
        )
        
        print(clf_res)
        print()
        
    model = GraphMAE(
        in_dim = feat_dim,
        emb_dim = config.emb_dim,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    
    def train_epoch():
        model.train() 
        
        loss = model.calc_loss(g=graph, feat=feat)
        
        return loss 
    
    
    def eval_epoch(mask: FloatArrayTensor) -> float:
        model.eval() 
        
        with torch.no_grad():
            emb = model.encode(g=graph, feat=feat)

        eval_acc = mlp_multiclass_classification(
            feat = emb,
            label = label,
            train_mask = train_mask,
            val_mask = mask,
        )['val_f1_micro']

        return eval_acc 
    
    
    recorder = ClassificationRecorder(model=model) 
    
    for epoch in range(1, config.num_epochs + 1):
        loss = train_epoch()
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
        recorder.train(epoch=epoch, loss=loss)

        if epoch % 100 == 0:
            val_acc = eval_epoch(mask=val_mask)
            recorder.validate(epoch=epoch, val_acc=val_acc)

    recorder.load_best_model_state()
    test_acc = eval_epoch(mask=test_mask)
    recorder.test(test_acc=test_acc)
        

if __name__ == '__main__':
    main() 
