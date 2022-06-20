from .imports import * 
from .metric import * 
from .util import * 
from sklearn.neighbors import KNeighborsClassifier

__all__ = [
    'KNeighbors_multiclass_classification',
    'xgb_multiclass_classification',
    'mlp_multilabel_classification',
]


def mlp_multilabel_classification(
    feat: FloatArray,
    label: BoolArray,
    train_mask: BoolArray,    
    val_mask: BoolArray,    
    test_mask: Optional[BoolArray] = None,
    use_gpu: bool = True,
    lr: float = 0.001,
    early_stopping_epochs: int = 50,
    use_tqdm: bool = True,
) -> dict[str, Any]:
    if use_gpu:
        device = get_device()
    else:
        device = torch.device('cpu')

    label = label.astype(np.int64)
    feat_th = torch.from_numpy(feat).to(torch.float32).to(device)
    label_th = torch.from_numpy(label).to(torch.float32).to(device)
    
    in_dim = feat.shape[-1]
    out_dim = label.shape[-1]
    hidden_dim = (in_dim + out_dim) // 2 
        
    model = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim), 
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_dict = {
        'epoch': 0,
        'val_f1_micro': 0.,
        'val_f1_macro': 0.,
        'test_f1_micro': 0.,
        'test_f1_macro': 0.,
    }
    
    for epoch in tqdm(itertools.count(1), disable=not use_tqdm, desc='mlp_multilabel_classification', unit='epoch'):
        model.train() 
        
        logits = model(feat_th[train_mask])
        
        loss = F.binary_cross_entropy_with_logits(input=logits, target=label_th[train_mask])
        
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        
        
        model.eval() 
        
        with torch.no_grad():
            logits = model(feat_th[val_mask])

        y_pred = (logits.detach().cpu().numpy() > 0).astype(np.int64)
        y_true = label[val_mask]
        val_f1_micro = calc_f1_micro(y_pred=y_pred, y_true=y_true)
        val_f1_macro = calc_f1_macro(y_pred=y_pred, y_true=y_true)
        
        if test_mask is not None:
            with torch.no_grad():
                logits = model(feat_th[test_mask])

            y_pred = (logits.detach().cpu().numpy() > 0).astype(np.int64)
            y_true = label[test_mask]
            test_f1_micro = calc_f1_micro(y_pred=y_pred, y_true=y_true)
            test_f1_macro = calc_f1_macro(y_pred=y_pred, y_true=y_true)
        else:
            test_f1_micro = test_f1_macro = 0.

        if val_f1_micro > best_dict['val_f1_micro']:
            best_dict['epoch'] = epoch 
            best_dict['val_f1_micro'] = val_f1_micro
            best_dict['val_f1_macro'] = val_f1_macro 
            best_dict['test_f1_micro'] = test_f1_micro 
            best_dict['test_f1_macro'] = test_f1_macro 

        if epoch - best_dict['epoch'] > early_stopping_epochs:
            break 
        
    return best_dict 


def KNeighbors_multiclass_classification(
    feat: FloatArray,
    label: IntArray,
    train_mask: BoolArray,    
    val_mask: BoolArray,    
    test_mask: Optional[BoolArray] = None,
    num_neighbors: int = 1,
) -> tuple[float, float, Optional[float], Optional[float]]:
    assert np.min(label) == 0 

    if test_mask is None:
        assert len(feat) == len(label) == len(train_mask) == len(val_mask)
        assert np.all(train_mask | val_mask)
        assert np.all(~(train_mask & val_mask))
    else:
        assert len(feat) == len(label) == len(train_mask) == len(val_mask) == len(test_mask)
        assert np.all(train_mask | val_mask | test_mask)
        assert np.all(~(train_mask & val_mask & test_mask))
    
    feat_dim = feat.shape[-1]
    num_classes = np.max(label) + 1 
    
    clf = KNeighborsClassifier(n_neighbors=num_neighbors)
    clf.fit(feat[train_mask], label[train_mask])

    val_y_pred = clf.predict(feat[val_mask])
    if test_mask is not None:
        test_y_pred = clf.predict(feat[test_mask])
    
    val_f1_micro = calc_f1_micro(y_pred=val_y_pred, y_true=label[val_mask])
    val_f1_macro = calc_f1_macro(y_pred=val_y_pred, y_true=label[val_mask])
    if test_mask is not None:
        test_f1_micro = calc_f1_micro(y_pred=test_y_pred, y_true=label[test_mask])
        test_f1_macro = calc_f1_macro(y_pred=test_y_pred, y_true=label[test_mask])
    
    if test_mask is None:
        return val_f1_micro, val_f1_macro, None, None 
    else:
        return val_f1_micro, val_f1_macro, test_f1_micro, test_f1_macro 


def xgb_multiclass_classification(
    feat: FloatArray,
    label: IntArray,
    train_mask: BoolArray,    
    val_mask: BoolArray,    
    test_mask: Optional[BoolArray] = None,
    verbose: bool = True, 
    check_mask: bool = True,
) -> dict[str, float]:
    assert np.min(label) == 0 

    if test_mask is None:
        assert len(feat) == len(label) == len(train_mask) == len(val_mask)
        
        if check_mask:
            assert np.all(train_mask | val_mask)
            assert np.all(~(train_mask & val_mask))
    else:
        assert len(feat) == len(label) == len(train_mask) == len(val_mask) == len(test_mask)
        
        if check_mask:
            assert np.all(train_mask | val_mask | test_mask)
            assert np.all(~(train_mask & val_mask & test_mask))
    
    feat_dim = feat.shape[-1]
    num_classes = np.max(label) + 1 
    
    xg_train = xgb.DMatrix(feat[train_mask], label=label[train_mask])
    xg_val = xgb.DMatrix(feat[val_mask], label=label[val_mask])
    if test_mask is not None:
        xg_test = xgb.DMatrix(feat[test_mask], label=label[test_mask])
    
    param = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 6, 
        'nthread': 4, 
        'num_class': num_classes, 
    }

    if test_mask is None:
        watch_list = [(xg_train, 'train'), (xg_val, 'val')]
    else:
        watch_list = [(xg_train, 'train'), (xg_val, 'val'), (xg_test, 'test')]
        
    num_round = 5
    
    if verbose:
        bst = xgb.train(param, xg_train, num_round, watch_list)
    else:
        bst = xgb.train(param, xg_train, num_round)
    
    val_y_pred = bst.predict(xg_val).astype(np.int64)
    if test_mask is not None:
        test_y_pred = bst.predict(xg_test).astype(np.int64)
    
    val_f1_micro = calc_f1_micro(y_pred=val_y_pred, y_true=label[val_mask])
    val_f1_macro = calc_f1_macro(y_pred=val_y_pred, y_true=label[val_mask])
    if test_mask is not None:
        test_f1_micro = calc_f1_micro(y_pred=test_y_pred, y_true=label[test_mask])
        test_f1_macro = calc_f1_macro(y_pred=test_y_pred, y_true=label[test_mask])
    else:
        test_f1_micro = test_f1_macro = 0.
    
    return {
        'val_f1_micro': val_f1_micro,
        'val_f1_macro': val_f1_macro,
        'test_f1_micro': test_f1_micro,
        'test_f1_macro': test_f1_macro,
    }
