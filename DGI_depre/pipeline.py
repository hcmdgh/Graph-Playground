from .model import * 
from .util import * 
from util import * 


class DGI_Pipeline:
    @classmethod
    def run(
        cls,
        homo_graph_path: str, 
        emb_dim: int = 512,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        init_log()
        device = auto_set_device()
        
        homo_graph = HomoGraph.load_from_file(homo_graph_path)
        
        g = homo_graph.to_dgl().to(device)
        
        # adj_mat = g.adj().to_dense().to(device)
        # adj_mat_np = adj_mat.cpu().numpy() 
        
        # adj_mat_np = normalize_adj_mat(adj_mat_np + np.eye(len(adj_mat_np)))
        # adj_mat_np += np.eye(len(adj_mat_np))
        
        # feat_np = homo_graph.node_attr_dict['feat'].cpu().numpy() 
        # feat_np = normalize_feature(feat_np)

        feat = homo_graph.node_attr_dict['feat'].to(device)
        feat_np = feat.cpu().numpy() 
        feat_dim = feat.shape[-1]
        
        label = homo_graph.node_attr_dict['label'].to(device)
        label_np = label.cpu().numpy() 

        num_nodes = homo_graph.num_nodes
        
        train_mask = homo_graph.node_attr_dict['train_mask'].cpu().numpy()
        val_mask = homo_graph.node_attr_dict['val_mask'].cpu().numpy()
        test_mask = homo_graph.node_attr_dict['test_mask'].cpu().numpy()
        
        model = DGI(in_dim=feat_dim, out_dim=emb_dim)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        early_stopping = EarlyStopping(
            monitor_field = 'loss',
            tolerance_epochs = 20,
            expected_trend = 'desc',
        )
        
        for epoch in itertools.count(1):
            model.train() 
            
            shuffled_idxs = np.random.permutation(num_nodes)
            
            shuffled_feat = feat[shuffled_idxs]
            
            zeros = torch.zeros(num_nodes).to(device)
            ones = torch.ones(num_nodes).to(device)
            discriminator_label = torch.cat([ones, zeros], dim=0)
            
            out_T, out_F = model(
                g = g,
                feat_T = feat,
                feat_F = shuffled_feat,
            )
            
            logits = torch.cat([out_T, out_F], dim=0)
            
            loss = F.binary_cross_entropy_with_logits(input=logits, target=discriminator_label)
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            
            early_stopping.record(
                epoch = epoch,
                loss = loss,
                model_state = copy.deepcopy(model.state_dict()),
            )
            
            logging.info(f"epoch: {epoch}, loss: {float(loss):.4f}")

            best_result = early_stopping.check_stop()
            
            if best_result:
                best_model_state = best_result['model_state']
                
                model.load_state_dict(best_model_state)
                
                break 
            
        val_f1_micro, val_f1_macro, test_f1_micro, test_f1_macro = xgb_multiclass_classification(
            feat = feat_np,
            label = label_np,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
            check_mask = False,
        )
        
        logging.info(f"XGBoost - raw feat: val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}, test_f1_micro: {test_f1_micro:.4f}, test_f1_macro: {test_f1_macro:.4f}")
            
        embedding = model.get_embedding(g=g, feat=feat)
        
        val_f1_micro, val_f1_macro, test_f1_micro, test_f1_macro = xgb_multiclass_classification(
            feat = embedding.cpu().numpy(),
            label = label_np,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
            check_mask = False,
        )
        
        logging.info(f"XGBoost - embedding: val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}, test_f1_micro: {test_f1_micro:.4f}, test_f1_macro: {test_f1_macro:.4f}")
