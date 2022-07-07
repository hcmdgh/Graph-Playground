from config import * 
from util import * 

import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import scipy.sparse as sp
from models import GMI, LogReg
from utils import process
from dl import * 


def main(config: Config):
    set_cwd(__file__)
    device = auto_set_device()
    init_log()
    
    graph = config.graph.to(device)
    feat = graph.ndata.pop('feat')
    label = graph.ndata.pop('label')
    num_classes = int(torch.max(label)) + 1 
    train_mask = graph.ndata.pop('train_mask')
    val_mask = graph.ndata.pop('val_mask')
    test_mask = graph.ndata.pop('test_mask')

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    adj_mat = graph.adj().to_dense().cpu().numpy() 

    adj_ori = sp.csr_matrix(adj_mat)
    features = sp.lil_matrix(feat.cpu().numpy())
    labels = np.eye(num_classes)[label.cpu().numpy()] 

    # adj_ori, features, labels, idx_train, idx_val, idx_test = process.load_data(args.dataset)
    # features, _ = process.preprocess_features(features)
    features = normalize_feature(feat)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]
    # adj = process.normalize_adj(adj_ori)
    adj = normalize_adj_mat(adj_mat)
    adj = sp.coo_matrix(adj) 

    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    # features = torch.FloatTensor(features[np.newaxis])
    features = features[None]
    labels = torch.FloatTensor(labels[np.newaxis])
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    model = GMI(ft_size, config.hidden_dim, config.gcn_act)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if torch.cuda.is_available():
        print('GPU available: Using CUDA')
        model.cuda()
        features = features.cuda()
        sp_adj = sp_adj.cuda()
        labels = labels.cuda()

    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    adj_dense = adj_ori.toarray()
    adj_target = adj_dense+np.eye(adj_dense.shape[0])
    adj_row_avg = 1.0/np.sum(adj_dense, axis=1)
    adj_row_avg[np.isnan(adj_row_avg)] = 0.0
    adj_row_avg[np.isinf(adj_row_avg)] = 0.0
    adj_dense = adj_dense*1.0
    for i in range(adj_ori.shape[0]):
        adj_dense[i] = adj_dense[i]*adj_row_avg[i]
    adj_ori = sp.csr_matrix(adj_dense, dtype=np.float32)


    for epoch in range(config.num_epochs):
        model.train()
        optimiser.zero_grad()
        
        res = model(features, adj_ori, config.num_negative_samples, sp_adj, None, None) 

        loss = config.alpha * process.mi_loss_jsd(res[0], res[1]) + config.beta * process.mi_loss_jsd(res[2], res[3]) + config.gamma * process.reconstruct_loss(res[4], adj_target)
        print('Epoch:', (epoch+1), '  Loss:', loss)

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_gmi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == 20:
            print('Early stopping!')
            break

        loss.backward()
        optimiser.step()

    print('Loading {}th epoch'.format(best_t+1))
    model.load_state_dict(torch.load('best_gmi.pkl'))

    embeds = model.embed(features, sp_adj)
    train_embs = embeds[0, train_mask]
    # val_embs = embeds[0, idx_val]      # typically, you could use the validation set
    test_embs = embeds[0, test_mask]

    train_lbls = torch.argmax(labels[0, train_mask], dim=1)
    # val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, test_mask], dim=1)

    accs = []

    iter_num = process.find_epoch(config.hidden_dim, nb_classes, train_embs, train_lbls, test_embs, test_lbls)
    for _ in range(50): 
        log = LogReg(config.hidden_dim, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0.00001)
        log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        for _ in range(iter_num):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            
            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        print(acc * 100)
        accs.append(acc * 100)

    accs = torch.stack(accs)
    print('Average accuracy:', accs.mean())
    print('STD:', accs.std())


if __name__ == '__main__':
    main(Config.cora_config())
