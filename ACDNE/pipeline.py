from util import * 
from .config import * 
from .util import * 
from .model import * 

__all__ = ['ACDNE_pipeline']


def ACDNE_pipeline(
    graph_S: dgl.DGLGraph,
    graph_T: dgl.DGLGraph,
):
    init_log()
    device = auto_set_device()
    
    feat_S = torch.tensor(graph_S.ndata['feat'], dtype=torch.float32, device=device)
    feat_S_np = feat_S.cpu().numpy()
    feat_T = torch.tensor(graph_T.ndata['feat'], dtype=torch.float32, device=device)
    feat_T_np = feat_T.cpu().numpy()

    N_S = len(feat_S_np)
    N_T = len(feat_T_np)
    
    feat_dim = feat_S_np.shape[-1]
    
    label_S = torch.tensor(graph_S.ndata['label'], dtype=torch.int64, device=device)
    label_S_np = label_S.cpu().numpy() 
    label_T = torch.tensor(graph_T.ndata['label'], dtype=torch.int64, device=device)
    label_T_np = label_T.cpu().numpy() 
    
    num_classes = np.max(label_S_np) + 1
    assert np.min(label_S_np) == np.min(label_T_np) == 0 
    assert np.max(label_S_np) == np.max(label_T_np)
    
    adj_mat_S = graph_S.adj().to_dense().numpy() 
    adj_mat_T = graph_T.adj().to_dense().numpy() 
    
    PPMI_S = calc_PPMI_mat(aggr_adj_mat(adj_mat_S))
    PPMI_T = calc_PPMI_mat(aggr_adj_mat(adj_mat_T))
    PPMI_norm_S = norm_adj_mat(PPMI_S)
    PPMI_norm_T = norm_adj_mat(PPMI_T)
    
    feat_neigh_S = PPMI_norm_S @ feat_S_np
    feat_neigh_T = PPMI_norm_T @ feat_T_np
    
    # [BEGIN] Visualization
    # label_S_arbitrary = torch.argmax(label_S, dim=-1).cpu()
    # label_T_arbitrary = torch.argmax(label_T_th, dim=-1).cpu()
    
    # g_S.ndata['label'] = label_S_arbitrary
    # g_T.ndata['label'] = label_T_arbitrary
    
    # nx.write_gexf(dgl.to_networkx(g_S, node_attrs=['label']), './ACDNE/output/citationv1.gexf')
    # nx.write_gexf(dgl.to_networkx(g_T, node_attrs=['label']), './ACDNE/output/dblpv7.gexf')

    # exit() 
    # [END]
    
    # [BEGIN] 合并源图和目标图    
    # g_S.ndata['feat'] = graph_S.node_prop_dict['feat']
    # g_S.ndata['label'] = graph_S.node_prop_dict['label'].to(torch.float32)
    # g_T.ndata['feat'] = graph_T.node_prop_dict['feat'] 
    # g_T.ndata['label'] = graph_T.node_prop_dict['label'].to(torch.float32)
    
    # combined_g = combine_graphs(src_g=g_S, tgt_g=g_T, add_super_node=True)
    # combined_g = to_device(combined_g)
    
    # combined_feat = combined_g.ndata['feat']
    # combined_label = combined_g.ndata['label']
    # combined_label_np = combined_g.ndata['label'].cpu().numpy() 
    
    # train_mask = np.zeros(combined_g.num_nodes(), dtype=bool)
    # val_mask = np.zeros(combined_g.num_nodes(), dtype=bool)
    # origin = combined_g.ndata['origin'].cpu().numpy() 
    # train_mask[origin == 1] = True 
    # val_mask[origin == 2] = True 
    # assert np.sum(train_mask) + np.sum(val_mask) == combined_g.num_nodes() - 2
    # [END]
    
    model = ACDNE(
        in_dim = feat_dim,
        num_classes = num_classes,
        dropout = 0.1,
    )
    
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    early_stopping = EarlyStopping(
        monitor_field = 'val_f1_micro',
        tolerance_epochs = EARLY_STOPPING_EPOCHS,
        expected_trend = 'asc', 
    )
    
    for epoch in itertools.count(1):
        # [BEGIN] Train
        model.train() 
        
        clf_loss_list = [] 
        domain_loss_list = [] 
        loss_list = [] 
        
        for step, (batch_idxs_S, batch_idxs_T) in enumerate(double_dataloader(N_S,
                                                                              N_T,
                                                                              batch_size=BATCH_SIZE)):
            batch_feat_S = feat_S_np[batch_idxs_S]
            batch_feat_neigh_S = feat_neigh_S[batch_idxs_S]
            batch_label_S = label_S_np[batch_idxs_S]
            batch_PPMI_S = PPMI_S[batch_idxs_S]
            batch_label_S_th = torch.from_numpy(batch_label_S).to(device)
            
            batch_feat_T = feat_T_np[batch_idxs_T]
            batch_feat_neigh_T = feat_neigh_T[batch_idxs_T]
            batch_label_T = label_T_np[batch_idxs_T]
            batch_PPMI_T = PPMI_T[batch_idxs_T]
            batch_label_T_th = torch.from_numpy(batch_label_T).to(device)

            batch_feat_ST = np.concatenate([batch_feat_S, batch_feat_T], axis=0)
            batch_feat_neigh_ST = np.concatenate([batch_feat_neigh_S, batch_feat_neigh_T], axis=0)
            batch_feat_ST_th = torch.from_numpy(batch_feat_ST).to(device)
            batch_feat_neigh_ST_th = torch.from_numpy(batch_feat_neigh_ST).to(device)

            domain_label = torch.tensor([1] * BATCH_SIZE + [0] * BATCH_SIZE, dtype=torch.int64, device=device)
            
            network_emb, pred_logits, domain_logits = model(
                feat_self = batch_feat_ST_th,
                feat_neigh = batch_feat_neigh_ST_th,
            )
            
            assert len(pred_logits) == len(domain_logits) == BATCH_SIZE * 2 
            pred_logits_S = pred_logits[:BATCH_SIZE]
            
            clf_loss = F.cross_entropy(input=pred_logits_S, target=batch_label_S_th)
            domain_loss = F.cross_entropy(input=domain_logits, target=domain_label)
            loss = clf_loss + domain_loss 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            clf_loss_list.append(float(clf_loss))
            domain_loss_list.append(float(domain_loss))
            loss_list.append(float(loss))
        
        logging.info(f"epoch: {epoch}, loss: {np.mean(loss_list):.4f}, clf_loss: {np.mean(clf_loss_list):.4f}, domain_loss: {np.mean(domain_loss_list):.4f}")
        # [END]
        
        # [BEGIN] Evaluate
        model.eval() 
        
        y_pred_list = [] 
        y_true_list = [] 
        
        for step, (batch_idxs_S, batch_idxs_T) in tqdm(enumerate(double_dataloader(N_S,
                                                                                   N_T,
                                                                                   batch_size=BATCH_SIZE))):
            batch_feat_S = feat_S_np[batch_idxs_S]
            batch_feat_neigh_S = feat_neigh_S[batch_idxs_S]
            batch_label_S = label_S_np[batch_idxs_S]
            batch_PPMI_S = PPMI_S[batch_idxs_S]
            batch_label_S_th = torch.from_numpy(batch_label_S).to(torch.float32).to(device)
            
            batch_feat_T = feat_T_np[batch_idxs_T]
            batch_feat_neigh_T = feat_neigh_T[batch_idxs_T]
            batch_label_T = label_T_np[batch_idxs_T]
            batch_PPMI_T = PPMI_T[batch_idxs_T]
            batch_label_T_th = torch.from_numpy(batch_label_T).to(torch.float32).to(device)

            batch_feat_ST = np.concatenate([batch_feat_S, batch_feat_T], axis=0)
            batch_feat_neigh_ST = np.concatenate([batch_feat_neigh_S, batch_feat_neigh_T], axis=0)
            batch_feat_ST_th = torch.from_numpy(batch_feat_ST).to(device)
            batch_feat_neigh_ST_th = torch.from_numpy(batch_feat_neigh_ST).to(device)

            with torch.no_grad():
                network_emb, pred_logits, domain_logits = model(
                    feat_self = batch_feat_ST_th,
                    feat_neigh = batch_feat_neigh_ST_th,
                )
            
            assert len(pred_logits) == len(domain_logits) == BATCH_SIZE * 2 
            pred_logits_S = pred_logits[:BATCH_SIZE].detach().cpu().numpy() 
            pred_logits_T = pred_logits[BATCH_SIZE:].detach().cpu().numpy() 
            
            y_pred = np.argmax(pred_logits_T, axis=-1) 
            
            y_pred_list.append(y_pred)
            y_true_list.append(batch_label_T)            

        y_pred = np.concatenate(y_pred_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)

        val_f1_micro = calc_f1_micro(y_true=y_true, y_pred=y_pred)
        val_f1_macro = calc_f1_macro(y_true=y_true, y_pred=y_pred)

        logging.info(f"epoch: {epoch}, val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}")

        early_stopping.record(
            epoch = epoch, 
            val_f1_micro = val_f1_micro,
            val_f1_macro = val_f1_macro,
        )
        
        early_stopping.auto_stop()
        # [END]
