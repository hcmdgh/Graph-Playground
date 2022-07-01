from .model.dgi import DGI 
from .model.logreg import LogReg
from .util import process
from .util import aug 

from util import * 


def GraphCL_pipeline(
    graph: dgl.DGLGraph,
    aug_type: str = 'subgraph',
    drop_ratio: float = 0.2,
    seed: int = 39,
    model_save_path: str = './GraphCL_official/output/model_state.pt',
):
    init_log()
    device = auto_set_device()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # training params

    batch_size = 1
    nb_epochs = 10000
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    drop_prob = 0.0
    hid_units = 512
    sparse = True

    nonlinearity = 'prelu' # special name to separate parameters
    
    # adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    num_nodes = graph.num_nodes() 
    
    edge_index = graph.edges() 
    adj_coo = sp.coo_matrix((np.ones(len(edge_index[0]), dtype=np.int64), (edge_index[0].numpy(), edge_index[1].numpy())), shape=[num_nodes, num_nodes], dtype=np.int64)
    adj = adj_coo.tocsr() 
    
    feat = graph.ndata['feat']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    label = graph.ndata['label']
    
    feat = process.preprocess_features(feat.numpy())

    feat_dim = feat.shape[1]   # node features dim
    num_classes = int(torch.max(label)) + 1

    feat = torch.from_numpy(feat[np.newaxis])

    '''
    ------------------------------------------------------------
    edge node mask subgraph
    ------------------------------------------------------------
    '''
    print("Begin Aug:[{}]".format(aug_type))
    if aug_type == 'edge':
        aug_features1 = feat
        aug_features2 = feat

        aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_ratio) # random drop edges
        aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_ratio) # random drop edges
        
    elif aug_type == 'node':
        aug_features1, aug_adj1 = aug.aug_drop_node(feat, adj, drop_percent=drop_ratio)
        aug_features2, aug_adj2 = aug.aug_drop_node(feat, adj, drop_percent=drop_ratio)
        
    elif aug_type == 'subgraph':
        aug_features1, aug_adj1 = aug.aug_subgraph(feat, adj, drop_percent=drop_ratio)
        aug_features2, aug_adj2 = aug.aug_subgraph(feat, adj, drop_percent=drop_ratio)

    elif aug_type == 'mask':
        aug_features1 = aug.aug_random_mask(feat,  drop_percent=drop_ratio)
        aug_features2 = aug.aug_random_mask(feat,  drop_percent=drop_ratio)
        
        aug_adj1 = adj
        aug_adj2 = adj

    else:
        raise AssertionError

    '''
    ------------------------------------------------------------
    '''

    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
    aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

    if sparse:
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        sp_aug_adj1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1)
        sp_aug_adj2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2)

    else:
        adj = (adj + sp.eye(adj.shape[0])).todense()
        aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
        aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()


    '''
    ------------------------------------------------------------
    mask
    ------------------------------------------------------------
    '''

    '''
    ------------------------------------------------------------
    '''
    if not sparse:
        adj = torch.FloatTensor(adj[np.newaxis])
        aug_adj1 = torch.FloatTensor(aug_adj1[np.newaxis])
        aug_adj2 = torch.FloatTensor(aug_adj2[np.newaxis])

    model = DGI(feat_dim, hid_units, nonlinearity)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    model.to(device)
    feat = feat.to(device)
    aug_features1 = aug_features1.to(device)
    aug_features2 = aug_features2.to(device)
    if sparse:
        sp_adj = sp_adj.to(device)
        sp_aug_adj1 = sp_aug_adj1.to(device)
        sp_aug_adj2 = sp_aug_adj2.to(device)
    else:
        adj = adj.to(device)
        aug_adj1 = aug_adj1.to(device)
        aug_adj2 = aug_adj2.to(device)

    label = label.to(device)

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):
        model.train()
        optimiser.zero_grad()

        idx = np.random.permutation(num_nodes)
        shuf_fts = feat[:, idx, :]

        lbl_1 = torch.ones(batch_size, num_nodes)
        lbl_2 = torch.zeros(batch_size, num_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)

        shuf_fts = shuf_fts.to(device)
        lbl = lbl.to(device)
        
        logits = model(feat, shuf_fts, aug_features1, aug_features2,
                    sp_adj if sparse else adj, 
                    sp_aug_adj1 if sparse else aug_adj1,
                    sp_aug_adj2 if sparse else aug_adj2,  
                    sparse, None, None, None, aug_type=aug_type) 

        loss = b_xent(logits, lbl)
        print('Loss:[{:.4f}]'.format(loss.item()))

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            
            torch.save(model.state_dict(), model_save_path)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(model_save_path))

    embeds, _ = model.embed(feat, sp_adj if sparse else adj, sparse, None)
    train_embs = embeds[0, train_mask]
    val_embs = embeds[0, val_mask]
    test_embs = embeds[0, test_mask]

    train_label = label[train_mask]
    val_label = label[val_mask]
    test_label = label[test_mask]

    tot = torch.zeros(1)
    tot = tot.cuda()

    accs = []

    for _ in range(50):
        log = LogReg(hid_units, num_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_label)
            
            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_label).float() / test_label.shape[0]
        accs.append(acc * 100)
        print('acc:[{:.4f}]'.format(acc))
        tot += acc

    print('-' * 100)
    print('Average accuracy:[{:.4f}]'.format(tot.item() / 50))
    accs = torch.stack(accs)
    print('Mean:[{:.4f}]'.format(accs.mean().item()))
    print('Std :[{:.4f}]'.format(accs.std().item()))
    print('-' * 100)
