from .imports import * 
from .metric import * 


def xgb_multiclass_classification(
    feat: FloatArray,
    label: IntArray,
    train_mask: BoolArray,    
    val_mask: BoolArray,    
    test_mask: Optional[BoolArray] = None,
    verbose: bool = True, 
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
    
    if test_mask is None:
        return val_f1_micro, val_f1_macro, None, None 
    else:
        return val_f1_micro, val_f1_macro, test_f1_micro, test_f1_macro 
