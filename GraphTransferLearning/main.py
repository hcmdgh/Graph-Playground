from util import * 
from dataset.AdaGCN_dataset import * 
from dataset.ACDNE_dataset import * 
from .config import * 
from .model.generator import * 
from .model.discriminator import * 
from .util import * 


def main():
    set_device(DEVICE)
    init_log('./log.log')
    
    src_dataset = load_AdaGCN_dataset(SRC_DATASET)
    tgt_dataset = load_AdaGCN_dataset(TGT_DATASET)
    src_g = src_dataset.to_dgl()
    tgt_g = tgt_dataset.to_dgl()
    
    feat_dim = src_dataset.node_prop_dict['feat'].shape[-1]
    num_classes = src_dataset.num_classes
    
    # [BEGIN] 合并源图和目标图    
    src_g.ndata['feat'] = src_dataset.node_prop_dict['feat']
    src_g.ndata['label'] = src_dataset.node_prop_dict['label'].to(torch.float32)
    tgt_g.ndata['feat'] = tgt_dataset.node_prop_dict['feat'] 
    tgt_g.ndata['label'] = tgt_dataset.node_prop_dict['label'].to(torch.float32)
    
    combined_g = combine_graphs(src_g=src_g, tgt_g=tgt_g, add_super_node=False)
    combined_g = to_device(combined_g)
    
    combined_feat = combined_g.ndata['feat']
    combined_label = combined_g.ndata['label']
    combined_label_np = combined_g.ndata['label'].cpu().numpy() 
    
    train_mask = np.zeros(combined_g.num_nodes(), dtype=bool)
    val_mask = np.zeros(combined_g.num_nodes(), dtype=bool)
    origin = combined_g.ndata['origin'].cpu().numpy() 
    train_mask[origin == 1] = True 
    val_mask[origin == 2] = True 
    assert np.sum(train_mask) + np.sum(val_mask) == combined_g.num_nodes() 
    # [END]
    
    generator = Generator(in_dim=feat_dim,
                          emb_dim=GNN_EMB_DIM,
                          num_layers=GNN_NUM_LAYERS,
                          num_classes=num_classes,
                          dropout=GNN_DROPOUT)
    discriminator = Discriminator(in_dim=GNN_EMB_DIM)
    generator = to_device(generator)
    discriminator = to_device(discriminator)
    
    optimizer_G = torch.optim.Adam(generator.parameters(), 
                                   lr=LR, 
                                   betas=(ADAM_B1, ADAM_B2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), 
                                   lr=LR, 
                                   betas=(ADAM_B1, ADAM_B2))  
    
    val_metric = MultiClassificationMetric(status='val')
    
    early_stopping = EarlyStopping(monitor_epochs=EARLY_STOPPING_EPOCH)
    
    for epoch in itertools.count(1):
        def train_epoch():
            generator.train()
            discriminator.train() 
            
            # [BEGIN] Train Generator
            gnn_emb, clf_out = generator(g=combined_g, 
                                         feat=combined_feat)

            out_D = discriminator(gnn_emb[train_mask])
            
            ones = to_device(torch.ones_like(out_D))
            
            pred_loss = F.binary_cross_entropy_with_logits(input=clf_out[train_mask], target=combined_label[train_mask])
            adversarial_loss = F.mse_loss(input=out_D, target=ones)

            if USE_GAN:
                loss_G = pred_loss + adversarial_loss
            else:
                loss_G = pred_loss
            
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step() 
            # [END]
            
            # [BEGIN] Train Discriminator
            if USE_GAN:
                with torch.no_grad():
                    gnn_emb, clf_out = generator(g=combined_g, 
                                                feat=combined_feat)
                    gnn_emb = gnn_emb.detach() 
                    
                real_out_D = discriminator(gnn_emb[train_mask])
                ones = to_device(torch.ones_like(real_out_D))
                real_loss_D = F.mse_loss(input=real_out_D, target=ones)
                
                fake_out_D = discriminator(gnn_emb[val_mask])
                zeros = to_device(torch.zeros_like(fake_out_D))
                fake_loss_D = F.mse_loss(input=fake_out_D, target=zeros)

                loss_D = real_loss_D + fake_loss_D 
                
                optimizer_D.zero_grad()
                loss_D.backward() 
                optimizer_D.step()
            else:
                loss_D = 0.
            # [END]
            
            val_loss = F.binary_cross_entropy_with_logits(input=clf_out[val_mask], target=combined_label[val_mask])

            early_stopping.record_loss(val_loss)

            logging.info(f"epoch: {epoch}, loss_G: {float(loss_G):.4f}, loss_D: {float(loss_D):.4f}, val_loss: {float(val_loss):.4f}")  
            
        def val():
            generator.eval() 
            
            with torch.no_grad(): 
                gnn_emb, clf_out = generator(g=combined_g, 
                                             feat=combined_feat)
                
            y_true = combined_label_np[val_mask]
            
            y_pred = np.zeros_like(y_true, dtype=np.int64) 
            logits_np = clf_out.cpu().numpy()[val_mask]
            y_pred[logits_np > 0] = 1 
            
            val_metric.measure(epoch=epoch, y_true=y_true, y_pred=y_pred)  
            
        train_epoch()  
        
        if epoch % 1 == 0:
            val()

            early_stopping.check_stop() 
    
    
if __name__ == '__main__':
    main() 
