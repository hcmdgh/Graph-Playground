from .config import * 
from util import * 

import argparse
import os
import time

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn.functional as F

from .input_data import load_data
from .model import * 
from .preprocess import mask_test_edges, mask_test_edges_dgl, sparse_to_tuple, preprocess_graph

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='Variant Graph Auto Encoder')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', '-h1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', '-h2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--datasrc', '-s', type=str, default='dgl',
                    help='Dataset download from dgl Dataset or website.')
parser.add_argument('--dataset', '-d', type=str, default='cora', help='Dataset string.')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use.')
args = parser.parse_args()


# check device
# device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")
# device = "cpu"

# roc_means = []
# ap_means = []

def calc_loss_weight(adj_mat: FloatTensor):
    N = len(adj_mat)
    E = float(torch.sum(adj_mat)) 
    
    pos_weight = (N * N - E) / E 
    norm = N * N / ((N * N - E) * 2)

    weight_mask = adj_mat.view(-1).nonzero().squeeze(dim=-1) 
    weight_tensor = to_device(torch.ones(N * N)) 
    weight_tensor[weight_mask] = pos_weight

    return weight_tensor, norm


def get_acc(adj_rec, adj_label):
    labels_all = adj_label.view(-1).long()
    preds_all = (adj_rec > 0.5).view(-1).long()
    accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
    return accuracy


def get_scores(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    adj_rec = adj_rec.cpu()
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]].item()))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def main():
    set_device(DEVICE)
    
    if DATASET == 'cora':
        dataset = CoraGraphDataset(reverse_edge=False)
    elif DATASET == 'citeseer':
        dataset = CiteseerGraphDataset(reverse_edge=False)
    elif DATASET == 'pubmed':
        dataset = PubmedGraphDataset(reverse_edge=False)
    else:
        raise AssertionError 
    
    graph = dataset[0]
    graph = to_device(graph)

    feat = graph.ndata.pop('feat')
    feat_dim = feat.shape[-1]

    full_adj_mat = to_device(graph.adjacency_matrix().to_dense()) 

    # build test set with 10% positive links
    train_edge_idx, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_dgl(graph, full_adj_mat)

    # create train graph
    train_edge_idx = to_device(torch.tensor(train_edge_idx)) 
    train_graph = dgl.edge_subgraph(graph, train_edge_idx, relabel_nodes=False)
    train_graph = train_graph
    train_adj_mat = to_device(train_graph.adjacency_matrix().to_dense())

    # compute loss parameters
    weight_tensor, norm = calc_loss_weight(train_adj_mat)
    if not USE_LOSS_NORM:
        norm = 1. 

    vgae = VGAE(feat_dim, args.hidden1, args.hidden2)

    optimizer = optim.Adam(vgae.parameters(), lr=LR)

    early_stopping = EarlyStopping(
        monitor_field = 'val_roc',
        tolerance_epochs = 100, 
        expected_trend = 'asc',
    )
    
    for epoch in itertools.count(1):
        vgae.train()

        _ = vgae(graph, feat)
        recon_adj = _['recon_adj']
        mean = _['mean']
        log_std = _['log_std']

        # compute loss
        if USE_LOSS_WEIGHT:
            recon_loss = norm * F.binary_cross_entropy(recon_adj.view(-1), train_adj_mat.view(-1), weight=weight_tensor)
        else:
            recon_loss = norm * F.binary_cross_entropy(recon_adj.view(-1), train_adj_mat.view(-1))
            
        kl_loss = -0.5 / len(recon_adj) * (1 + 2 * log_std - mean ** 2 - torch.exp(log_std) ** 2).sum(dim=1).mean()

        if USE_KL_LOSS:
            loss = recon_loss + kl_loss
        else:
            loss = recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # train_acc = get_acc(recon_adj, train_adj_mat)
        if epoch % 1 == 0:
            val_roc, val_ap = get_scores(val_edges, val_edges_false, recon_adj)

            test_roc, test_ap = get_scores(test_edges, test_edges_false, recon_adj)

            print(f"epoch: {epoch}, train_loss: {float(loss):.4f}, val_roc: {val_roc:.4f}, val_ap: {val_ap:.4f}, test_roc: {test_roc:.4f}, test_ap: {test_ap:.4f}")

            early_stopping.record(
                epoch = epoch,
                val_roc = val_roc,
                val_ap = val_ap,
                test_roc = test_roc,
                test_ap = test_ap,
            )

            early_stopping.auto_stop()


if __name__ == '__main__':
    main()
