from util import * 
from .config import * 
from .model import * 
from dataset.Office31 import * 


def main():
    set_device(DEVICE)
    
    init_log('./log.log')
    
    src_dataloader = get_Office31_dataloader(
        dataset_name = SRC_DATASET,
        status = 'train', 
        batch_size = BATCH_SIZE,
    )
    
    tgt_dataloader = get_Office31_dataloader(
        dataset_name = TGT_DATASET,
        status = 'train', 
        batch_size = BATCH_SIZE,
    )
    
    tgt_eval_dataloader = get_Office31_dataloader(
        dataset_name = TGT_DATASET,
        status = 'eval', 
        batch_size = BATCH_SIZE,
    )
    
    model = DSAN(num_classes=NUM_CLASSES,
                 bottle_neck=True)
    model = to_device(model)
    
    # optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = torch.optim.SGD([
        {'params': model.resnet.parameters()},
        {'params': model.bottle_fc.parameters(), 'lr': LR[1]},
        {'params': model.clf_fc.parameters(), 'lr': LR[2]},
    ], lr=LR[0], momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    early_stopping = EarlyStopping(monitor_epochs=100)
    
    eval_metric = MultiClassificationMetric(status='val')
    
    for epoch in itertools.count(1):
        def train_epoch():
            model.train() 
            
            combined_dataloader = combine_dataloaders(src_dataloader, tgt_dataloader)

            loss_list = []
            lmmd_loss_list = [] 
            clf_loss_list = [] 
            
            for step, ((src_img_batch, src_label), (tgt_img_batch, tgt_label)) in enumerate(tqdm(combined_dataloader, desc='train')):
                src_img_batch = to_device(src_img_batch)
                tgt_img_batch = to_device(tgt_img_batch)
                src_label = to_device(src_label)
                tgt_label = to_device(tgt_label)

                src_pred, tgt_pred, lmmd_loss = model(
                    src_img_batch = src_img_batch,
                    tgt_img_batch = tgt_img_batch,
                    src_label = src_label, 
                )
                
                clf_loss = F.cross_entropy(input=src_pred, target=src_label)
                
                if not USE_LMMD_LOSS:
                    lmmd_loss = 0. 
                
                loss = clf_loss + LMMD_LOSS_WEIGHT * lmmd_loss
                
                optimizer.zero_grad() 
                loss.backward() 
                optimizer.step() 
            
                loss_list.append(float(loss))
                lmmd_loss_list.append(float(lmmd_loss))
                clf_loss_list.append(float(clf_loss))

            logging.info(f"epoch: {epoch}, loss: {np.mean(loss_list):.4f}, lmmd_loss: {np.mean(lmmd_loss_list):.4f}, clf_loss: {np.mean(clf_loss_list):.4f}") 

        def eval():
            model.eval() 
            
            eval_loss_list = [] 
            
            full_y_true = np.zeros([0], dtype=np.int64)
            full_y_pred = np.zeros([0], dtype=np.int64)
            
            with torch.no_grad():
                for tgt_img_batch, tgt_label in tqdm(tgt_eval_dataloader, desc='eval'):
                    tgt_img_batch = to_device(tgt_img_batch)
                    tgt_label = to_device(tgt_label)
                    
                    tgt_pred = model.predict(tgt_img_batch)

                    loss = F.cross_entropy(input=tgt_pred, target=tgt_label)
                    
                    eval_loss_list.append(float(loss))
                    
                    y_true_np = tgt_label.cpu().numpy() 
                    y_pred_np = tgt_pred.cpu().numpy() 
                    y_pred_np = np.argmax(y_pred_np, axis=-1)
                    
                    full_y_true = np.concatenate([full_y_true, y_true_np])
                    full_y_pred = np.concatenate([full_y_pred, y_pred_np])
                    
            eval_loss = np.mean(eval_loss_list)
            logging.info(f"epoch: {epoch}, eval_loss: {eval_loss:.4f}")

            eval_metric.measure(epoch=epoch, y_true=full_y_true, y_pred=full_y_pred)

            early_stopping.record_loss(eval_loss)
            
            early_stopping.check_stop()
            
        train_epoch() 
        
        eval()


if __name__ == '__main__':
    main() 
