from .pipeline import * 
dglnn.SAGEConv

if __name__ == '__main__':
    GraphSAGE_Pipeline.run_node_classification(
        homo_graph_path = '/home/gh/Dataset/GengHao/HomoGraph/DGL/Cora.pt',
        use_gpu = True,
        batch_norm = True,
        hidden_dim = 64,
        num_layers = 2,
        
        use_sampler = False,
        batch_size = 1024,
        
        manually_split_train_set = True,
        train_val_test_ratio = (0.6, 0.2, 0.2),
    )
