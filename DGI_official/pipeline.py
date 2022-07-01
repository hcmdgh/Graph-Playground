from .dgi import DGI, Classifier

from util import * 

__all__ = ['DGI_pipeline']


def DGI_pipeline(
    graph: dgl.DGLGraph,
    dropout: float = 0.,
    lr: float = 0.001,
    weight_decay: float = 0.,
    classifier_lr: float = 0.01,
    num_epochs: int = 300,
    num_classifier_epochs: int = 300,
    hidden_dim: int = 512,
    num_gcn_layers: int = 1,
    early_stopping_epochs: int = 20,
    add_self_loop: bool = True, 
    model_save_path: str = './DGI_official/output/model_state.pt',
):
    init_log()
    device = auto_set_device()
    
    graph = graph.to(device)    
    feat = graph.ndata['feat']
    label = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    
    feat_dim = feat.shape[-1]
    num_classes = int(torch.max(label)) + 1 

    if add_self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

    model = DGI(
        g = graph,
        in_feats = feat_dim,
        n_hidden = hidden_dim,
        n_layers = num_gcn_layers,
        activation = nn.PReLU(hidden_dim),
        dropout = dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(num_epochs):
        model.train()

        optimizer.zero_grad()
        loss = model(feat)
        loss.backward()
        optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            cnt_wait += 1

        if cnt_wait >= early_stopping_epochs:
            print('Early stopping!')
            break
        
        log_multi(
            epoch = epoch,
            loss = float(loss),
        )
        
    # model.load_state_dict(torch.load(model_save_path))

    # with torch.no_grad():
    #     emb = model.encoder(feat, corrupt=False)

    # emb = emb.detach().cpu().numpy() 
    
    # xgb_res = xgb_multiclass_classification(
    #     feat = emb,
    #     label = label.cpu().numpy(),
    #     train_mask = train_mask.cpu().numpy(),
    #     val_mask = val_mask.cpu().numpy(),
    #     test_mask = test_mask.cpu().numpy(),
    #     check_mask = False, 
    # )
    
    # print(xgb_res)

    classifier = Classifier(hidden_dim, num_classes).to(device)

    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=classifier_lr, weight_decay=weight_decay)

    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(model_save_path))
    embeds = model.encoder(feat, corrupt=False)
    embeds = embeds.detach()
    dur = []
    for epoch in range(num_classifier_epochs):
        classifier.train()

        classifier_optimizer.zero_grad()
        preds = classifier(embeds)
        loss = F.nll_loss(preds[train_mask], label[train_mask])
        loss.backward()
        classifier_optimizer.step()
        
        emb = embeds.detach().cpu()
        
        val_f1_micro = calc_f1_micro(y_pred=preds[val_mask], y_true=label[val_mask])
        val_f1_macro = calc_f1_macro(y_pred=preds[val_mask], y_true=label[val_mask])
        test_f1_micro = calc_f1_micro(y_pred=preds[test_mask], y_true=label[test_mask])
        test_f1_macro = calc_f1_macro(y_pred=preds[test_mask], y_true=label[test_mask])
        
        log_multi(
            epoch = epoch,
            loss = float(loss),
            val_f1_micro = val_f1_micro,
            val_f1_macro = val_f1_macro,
            test_f1_micro = test_f1_micro,
            test_f1_macro = test_f1_macro,
        )
