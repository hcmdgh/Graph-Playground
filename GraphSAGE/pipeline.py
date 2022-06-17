from util import * 
from .model import * 


class GraphSAGE_Pipeline:
    @classmethod
    def run_node_classification(
        cls,
        *,
        homo_graph_path: str,
        use_gpu: bool = True,
        add_self_loop: bool = False,
        to_bidirected: bool = True,
        use_sampler: bool = False,
        batch_size: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.5,
        batch_norm: bool = True,
        lr: float = 0.01,
        weight_decay: float = 0.,
        early_stopping_epochs: int = 50,
        manually_split_train_set: bool = False,
        train_val_test_ratio: Optional[tuple[float, float, float]] = None, 
    ):
        init_log('./log.log')
        device = auto_set_device(use_gpu=use_gpu)
        
        homo_graph = HomoGraph.load_from_file(homo_graph_path)
        g = homo_graph.to_dgl()
        
        feat = homo_graph.node_attr_dict['feat'].to(device)
        feat_dim = feat.shape[-1]
        label = homo_graph.node_attr_dict['label'].to(device)
        label_np = label.cpu().numpy()
        
        if not manually_split_train_set:
            train_mask = homo_graph.node_attr_dict['train_mask'].numpy()
            val_mask = homo_graph.node_attr_dict['val_mask'].numpy()
            test_mask = homo_graph.node_attr_dict['test_mask'].numpy()
        else:
            train_mask, val_mask, test_mask = split_train_val_test_set(
                total_cnt = homo_graph.num_nodes,
                train_ratio = train_val_test_ratio[0],
                val_ratio = train_val_test_ratio[1],
                test_ratio = train_val_test_ratio[2],
            )
        
        if to_bidirected:
            g = dgl.to_bidirected(g)
        
        if add_self_loop:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            
        g = g.to(device)
        
        model = GraphSAGE(
            in_dim = feat_dim,
            hidden_dim = hidden_dim,
            out_dim = homo_graph.num_classes,
            num_layers = num_layers,
            dropout = dropout,
            batch_norm = batch_norm,
        )
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        early_stopping = EarlyStopping(
            monitor_field = 'val_f1_micro',
            tolerance_epochs = early_stopping_epochs,
            expected_trend = 'asc', 
        )
        
        if use_sampler:
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=num_layers)
            
            dataloader = dgl.dataloading.DataLoader(
                graph = g,
                indices = torch.arange(g.num_nodes())[train_mask].to(device),
                graph_sampler = sampler,
                batch_size = batch_size,
                shuffle = True,
                drop_last = False,
                num_workers = 0, 
                device = device,
            )
        
        for epoch in itertools.count(1):
            model.train() 
            
            if not use_sampler:
                logits = model(g, feat)
                
                loss = F.cross_entropy(input=logits[train_mask], target=label[train_mask])
                
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
            
            else:
                loss_list = [] 
                
                for step, (in_nids, out_nids, blocks) in enumerate(dataloader):
                    feat_batch = feat[in_nids]
                    
                    logits = model.forward_batch(blocks, feat_batch)    

                    loss = F.cross_entropy(input=logits, target=label[out_nids])
                
                    optimizer.zero_grad()
                    loss.backward() 
                    optimizer.step() 

                    loss_list.append(float(loss))

                loss = np.mean(loss_list)
                
            model.eval() 
            
            with torch.no_grad():
                logits = model(g, feat)
                
            y_pred = logits.detach().cpu().numpy() 
            y_pred = np.argmax(y_pred, axis=-1)
            y_true = label_np 
            
            val_f1_micro = calc_f1_micro(y_pred=y_pred[val_mask], y_true=y_true[val_mask])
            val_f1_macro = calc_f1_macro(y_pred=y_pred[val_mask], y_true=y_true[val_mask])
            test_f1_micro = calc_f1_micro(y_pred=y_pred[test_mask], y_true=y_true[test_mask])
            test_f1_macro = calc_f1_macro(y_pred=y_pred[test_mask], y_true=y_true[test_mask])

            hidden_emb = model.get_hidden_embedding(g, feat)

            nmi, ari = KMeans_clustering_evaluate(
                feat = hidden_emb.cpu().numpy(),
                label = label_np,
                num_classes = homo_graph.num_classes, 
                num_runs = 2,
            )

            early_stopping.record(
                epoch = epoch,
                val_f1_micro = val_f1_micro,
                val_f1_macro = val_f1_macro,
                test_f1_micro = test_f1_micro,
                test_f1_macro = test_f1_macro,
                nmi = nmi,
                ari = ari,  
            )
            
            logging.info(f"epoch: {epoch}, loss: {float(loss):.4f}, val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}, test_f1_micro: {test_f1_micro:.4f}, test_f1_macro: {test_f1_macro:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}")
            
            if early_stopping.check_stop():
                break 
            
        print("\nUse XGBoost:")
        
        val_f1_micro, val_f1_macro, test_f1_micro, test_f1_macro = xgb_multiclass_classification(
            feat = feat.cpu().numpy(),
            label = label_np,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
        )

        print(f"val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}, test_f1_micro: {test_f1_micro:.4f}, test_f1_macro: {test_f1_macro:.4f}")

    @classmethod
    def run_node_clustering(
        cls,
        *,
        homo_graph_path: str,
        use_gpu: bool = True,
        add_self_loop: bool = False,
        to_bidirected: bool = True,
        use_sampler: bool = False,
        batch_size: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.5,
        batch_norm: bool = True,
        lr: float = 0.01,
        weight_decay: float = 0.,
        early_stopping_epochs: int = 50,
        manually_split_train_set: bool = False,
        train_val_test_ratio: Optional[tuple[float, float, float]] = None, 
    ):
        init_log('./log.log')
        device = auto_set_device(use_gpu=use_gpu)
        
        homo_graph = HomoGraph.load_from_file(homo_graph_path)
        g = homo_graph.to_dgl()
        
        feat = homo_graph.node_attr_dict['feat'].to(device)
        feat_dim = feat.shape[-1]
        label = homo_graph.node_attr_dict['label'].to(device)
        label_np = label.cpu().numpy()
        
        if not manually_split_train_set:
            train_mask = homo_graph.node_attr_dict['train_mask'].numpy()
            val_mask = homo_graph.node_attr_dict['val_mask'].numpy()
            test_mask = homo_graph.node_attr_dict['test_mask'].numpy()
        else:
            train_mask, val_mask, test_mask = split_train_val_test_set(
                total_cnt = homo_graph.num_nodes,
                train_ratio = train_val_test_ratio[0],
                val_ratio = train_val_test_ratio[1],
                test_ratio = train_val_test_ratio[2],
            )
        
        if to_bidirected:
            g = dgl.to_bidirected(g)
        
        if add_self_loop:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
            
        g = g.to(device)
        
        model = GraphSAGE(
            in_dim = feat_dim,
            hidden_dim = hidden_dim,
            out_dim = homo_graph.num_classes,
            num_layers = num_layers,
            dropout = dropout,
            batch_norm = batch_norm,
        )
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        early_stopping = EarlyStopping(
            monitor_field = 'nmi',
            tolerance_epochs = early_stopping_epochs,
            expected_trend = 'asc', 
        )
        
        if use_sampler:
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=num_layers)
            
            dataloader = dgl.dataloading.DataLoader(
                graph = g,
                indices = torch.arange(g.num_nodes())[train_mask].to(device),
                graph_sampler = sampler,
                batch_size = batch_size,
                shuffle = True,
                drop_last = False,
                num_workers = 0, 
                device = device,
            )
        
        for epoch in itertools.count(1):
            model.train() 
            
            if not use_sampler:
                logits = model(g, feat)
                
                loss = F.cross_entropy(input=logits[train_mask], target=label[train_mask])
                
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step() 
            
            else:
                loss_list = [] 
                
                for step, (in_nids, out_nids, blocks) in enumerate(dataloader):
                    feat_batch = feat[in_nids]
                    
                    logits = model.forward_batch(blocks, feat_batch)    

                    loss = F.cross_entropy(input=logits, target=label[out_nids])
                
                    optimizer.zero_grad()
                    loss.backward() 
                    optimizer.step() 

                    loss_list.append(float(loss))

                loss = np.mean(loss_list)
                
            model.eval() 
            
            with torch.no_grad():
                logits = model(g, feat)
                
            y_pred = logits.detach().cpu().numpy() 
            y_pred = np.argmax(y_pred, axis=-1)
            y_true = label_np 
            
            val_f1_micro = calc_f1_micro(y_pred=y_pred[val_mask], y_true=y_true[val_mask])
            val_f1_macro = calc_f1_macro(y_pred=y_pred[val_mask], y_true=y_true[val_mask])
            test_f1_micro = calc_f1_micro(y_pred=y_pred[test_mask], y_true=y_true[test_mask])
            test_f1_macro = calc_f1_macro(y_pred=y_pred[test_mask], y_true=y_true[test_mask])

            hidden_emb = model.get_hidden_embedding(g, feat)

            nmi, ari = KMeans_clustering_evaluate(
                feat = hidden_emb.cpu().numpy(),
                label = label_np,
                num_classes = homo_graph.num_classes, 
                num_runs = 2,
            )

            early_stopping.record(
                epoch = epoch,
                val_f1_micro = val_f1_micro,
                val_f1_macro = val_f1_macro,
                test_f1_micro = test_f1_micro,
                test_f1_macro = test_f1_macro,
                nmi = nmi,
                ari = ari,  
                model_state = copy.deepcopy(model.state_dict()), 
            )
            
            logging.info(f"epoch: {epoch}, loss: {float(loss):.4f}, val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}, test_f1_micro: {test_f1_micro:.4f}, test_f1_macro: {test_f1_macro:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}")
            
            best_result_dict = early_stopping.check_stop()

            if best_result_dict:
                torch.save(best_result_dict['model_state'], './GraphSAGE/output/model_state.pt')
                break 
            
        print("\nUse XGBoost:")
        
        val_f1_micro, val_f1_macro, test_f1_micro, test_f1_macro = xgb_multiclass_classification(
            feat = feat.cpu().numpy(),
            label = label_np,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
        )

        print(f"val_f1_micro: {val_f1_micro:.4f}, val_f1_macro: {val_f1_macro:.4f}, test_f1_micro: {test_f1_micro:.4f}, test_f1_macro: {test_f1_macro:.4f}")
