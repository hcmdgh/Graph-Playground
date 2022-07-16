from model import * 

from dl import * 
import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def main(args):
    set_cwd(__file__)
    device = auto_set_device()
    init_log()
    
    # load and preprocess dataset
    data = load_data(args)
    g = data[0]
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = g.number_of_edges()

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    g = g.to(device)
    # create DGI model
    dgi = DGI(g,
              in_feats,
              args.n_hidden,
              args.n_layers,
              nn.PReLU(args.n_hidden),
              args.dropout).to(device)

    dgi_optimizer = torch.optim.Adam(dgi.parameters(),
                                     lr=args.dgi_lr,
                                     weight_decay=args.weight_decay)

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    dur = []
    for epoch in range(args.n_dgi_epochs):
        dgi.train()
        if epoch >= 3:
            t0 = time.time()

        dgi_optimizer.zero_grad()
        loss = dgi(features)
        loss.backward()
        dgi_optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            n_edges / np.mean(dur) / 1000))

    # create classifier model
    classifier = Classifier(args.n_hidden, n_classes).to(device)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(),
                                            lr=args.classifier_lr,
                                            weight_decay=args.weight_decay)

    # train classifier
    print('Loading {}th epoch'.format(best_t))
    dgi.load_state_dict(torch.load('best_dgi.pkl'))
    embeds = dgi.encoder(features, corrupt=False)
    embeds = embeds.detach()
    dur = []
    for epoch in range(args.n_classifier_epochs):
        classifier.train()
        if epoch >= 3:
            t0 = time.time()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[train_mask], labels[train_mask])
        loss.backward()
        classifier_optimizer.step()
        
        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(classifier, embeds, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

    print()
    acc = evaluate(classifier, embeds, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGI')
    
    parser.add_argument("--dataset", type=str, default='cora')
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--dgi-lr", type=float, default=1e-3,
                        help="dgi learning rate")
    parser.add_argument("--classifier-lr", type=float, default=1e-2,
                        help="classifier learning rate")
    parser.add_argument("--n-dgi-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    
    main(args)
