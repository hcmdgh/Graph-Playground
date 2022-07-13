from util import * 
from .model import * 


class HeCo_Pipeline:
    @classmethod
    def run(
        cls,
        *,
        hetero_graph_path: str,
        positive_sample_mask_path: str, 
        use_gpu: bool = True,
        relation_neighbor_size_dict: dict[EdgeType, int],
        metapaths: list[list],
        emb_dim: int = 64,
        tau: float = 0.8,
        lambda_: float = 0.5,
        feat_dropout: float = 0.3,
        attn_dropout: float = 0.5,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        early_stopping_epochs: int = 50,
        lr: float = 0.0008,
    ):
        init_log('./log.log')
        device = auto_set_device(use_gpu=use_gpu)

        hetero_graph = HeteroGraph.load_from_file(hetero_graph_path)
        hg = hetero_graph.to_dgl()
        
        positive_sample_mask = np.load(positive_sample_mask_path)
        
        # [BEGIN] 没有初始特征的结点采用onehot编码
        feat_dict = dict() 
        
        for node_type in hetero_graph.node_types:
            if node_type in hetero_graph.node_attr_dict['feat']:
                feat_dict[node_type] = hetero_graph.node_attr_dict['feat'][node_type]
            else:
                feat_dict[node_type] = torch.eye(hetero_graph.num_nodes_dict[node_type])
        # [END]
        
        assert len(hetero_graph.node_attr_dict['label']) == 1
        infer_node_type = next(iter(hetero_graph.node_attr_dict['label']))
        
        label = hetero_graph.node_attr_dict['label'][infer_node_type]
        label_np = label.numpy() 
        
        # train_mask = hetero_graph.node_attr_dict['train_mask'][infer_node_type].numpy()
        # val_mask = hetero_graph.node_attr_dict['val_mask'][infer_node_type].numpy()
        # test_mask = hetero_graph.node_attr_dict['test_mask'][infer_node_type].numpy()
        
        train_mask, val_mask, test_mask = random_split_dataset(
            total_cnt = len(label),
            train_ratio = train_ratio,
            val_ratio = val_ratio,
            test_ratio = test_ratio,
            seed = 142857, 
        )

        model = HeCo(
            hg = hg,
            infer_node_type = infer_node_type,
            feat_dict = feat_dict,
            metapaths = metapaths,
            relation_neighbor_size_dict = relation_neighbor_size_dict,
            positive_sample_mask = positive_sample_mask, 
            emb_dim = emb_dim,
            tau = tau,
            lambda_ = lambda_,
            feat_dropout = feat_dropout,
            attn_dropout = attn_dropout, 
        )
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        early_stopping = EarlyStopping(
            monitor_field = 'val_f1_micro',
            expected_trend = 'asc',
            tolerance_epochs = early_stopping_epochs, 
        )
        
        for epoch in itertools.count(1):
            model.train() 
            
            loss = model()
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            
            logging.info(f"epoch: {epoch}, loss: {float(loss):.4f}")

            
            model.eval()
            
            emb_np = model.calc_node_embedding().cpu().numpy() 
            
            val_f1_micro, val_f1_macro, test_f1_micro, test_f1_macro = xgb_multiclass_classification(
                feat = emb_np,
                label = label_np,
                train_mask = train_mask,
                val_mask = val_mask,
                test_mask = test_mask, 
            )
            
            early_stopping.record(
                epoch = epoch,
                val_f1_micro = val_f1_micro, 
                val_f1_macro = val_f1_macro, 
                test_f1_micro = test_f1_micro, 
                test_f1_macro = test_f1_macro, 
            )
            
            logging.info(f"epoch: {epoch}, val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}, test_f1_micro: {test_f1_micro:.4f}, test_f1_macro: {test_f1_macro:.4f}")

            early_stopping.auto_stop()
