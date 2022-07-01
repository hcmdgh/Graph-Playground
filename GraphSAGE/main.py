from .pipeline import * 

if __name__ == '__main__':
    with open('./MAG/output/hg.pkl', 'rb') as fp:
        hg = pickle.load(fp)    

    dgl.save_graphs('/home/Dataset/GengHao/HeteroGraph/MAG/mag_cs.dgl', [hg])
    exit()

    g = hg['pp']
    
    GraphSAGE_Pipeline.run_node_classification(
        g = g,
        use_gpu = True,
        batch_norm = True,
        hidden_dim = 64,
        num_layers = 2,
        
        use_sampler = False,
        batch_size = 1024,
        
        manually_split_train_set = True,
        train_val_test_ratio = (0.6, 0.2, 0.2),
    )
