from .model import * 
from util import *


def main(
    homo_graph_path_S: str,
    homo_graph_path_T: str,
    batch_size: int = 64,
    lr: float = 0.001,
    num_epochs: int = 200,
    lmmd_weight: float = 0.5,
):
    init_log()
    device = auto_set_device()
    
    homo_graph_S = HomoGraph.load_from_file(homo_graph_path_S)
    homo_graph_T = HomoGraph.load_from_file(homo_graph_path_T)

    feat_S = homo_graph_S.node_attr_dict['feat'].to(device)
    feat_T = homo_graph_T.node_attr_dict['feat'].to(device)
    feat_S_np = feat_S.cpu().numpy()
    feat_T_np = feat_T.cpu().numpy()
    
    num_nodes_S = len(feat_S)
    num_nodes_T = len(feat_T)
    
    # [BEGIN] 标签重新编码
    label_S_raw = homo_graph_S.node_attr_dict['label'].numpy().astype(bool)
    label_T_raw = homo_graph_T.node_attr_dict['label'].numpy().astype(bool)

    label_map: dict[tuple, int] = dict() 
    
    for label in np.concatenate([label_S_raw, label_T_raw], axis=0):
        label = tuple(label)
        
        if label not in label_map:
            label_map[label] = len(label_map)
            
    num_classes = len(label_map)
    assert num_classes == 12 
    
    label_S_list = []
    label_T_list = [] 
    
    for label in label_S_raw:
        label = tuple(label)
        label_S_list.append(label_map[label])
        
    for label in label_T_raw:
        label = tuple(label)
        label_T_list.append(label_map[label])
        
    label_S_np = np.array(label_S_list, dtype=np.int64)
    label_T_np = np.array(label_T_list, dtype=np.int64)
    label_S = torch.from_numpy(label_S_np).to(device)
    label_T = torch.from_numpy(label_T_np).to(device)
    # [END]

    feat_dim = feat_S.shape[-1]

    # [BEGIN] 直接用原始数据进行分类实验
    if False:
        combined_feat = np.concatenate([feat_S_np, feat_T_np], axis=0)
        combined_label = np.concatenate([label_S_np, label_T_np], axis=0)
        
        train_mask = np.zeros(len(combined_feat), dtype=bool)
        train_mask[:len(feat_S_np)] = True 
        val_mask = ~train_mask 
        
        res_dict = xgb_multiclass_classification(
            feat = combined_feat,
            label = combined_label,
            train_mask = train_mask,
            val_mask = val_mask,
        )
        
        print("直接用原始数据进行分类实验：")
        print(res_dict)
        print()
    # [END]
    
    model = DSAN(
        in_dim = feat_dim,
        num_classes = num_classes,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, num_epochs + 1):
        model.train() 
        
        loss_list = []
        
        for idxs_S, idxs_T in double_dataloader(num_nodes_S, num_nodes_T, batch_size=batch_size):
            feat_batch_S = feat_S[idxs_S]
            feat_batch_T = feat_T[idxs_T]
            label_batch_S = label_S[idxs_S]
            label_batch_T = label_T[idxs_T]

            pred_S, pred_T, lmmd_loss = model(
                feat_batch_S = feat_batch_S,
                feat_batch_T = feat_batch_T,
                label_batch_S = label_batch_S,
            )
            
            loss = calc_loss(
                epoch = epoch,
                num_epochs = num_epochs,
                label_S = label_batch_S,
                pred_S = pred_S,
                lmmd_loss = lmmd_loss,
                lmmd_weight = lmmd_weight,
            )
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            loss_list.append(float(loss))
            
        if epoch % 1 == 0:
            with torch.no_grad():
                pred_T = model.predict(feat_T)

            y_pred = np.argmax(pred_T.detach().cpu().numpy(), axis=-1) 
                
            val_f1_micro = calc_f1_micro(y_pred=y_pred, y_true=label_T_np)
            val_f1_macro = calc_f1_macro(y_pred=y_pred, y_true=label_T_np)

            logging.info(f"epoch: {epoch}, loss: {np.mean(loss_list):.4f}, val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}")
            

if __name__ == '__main__':
    main(
        homo_graph_path_S = '/home/Dataset/GengHao/HomoGraph/ACDNE/acmv9.pt',
        homo_graph_path_T = '/home/Dataset/GengHao/HomoGraph/ACDNE/citationv1.pt',
    )
