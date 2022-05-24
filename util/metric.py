from .imports import * 
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


def calc_f1_micro(y_true: IntArray,
                  y_pred: IntArray) -> float:
    return f1_score(y_true=y_true, y_pred=y_pred, average='micro')


def calc_f1_macro(y_true: IntArray,
                  y_pred: IntArray) -> float:
    return f1_score(y_true=y_true, y_pred=y_pred, average='macro')


def calc_roc_auc_score(y_true: IntArray,
                       y_pred: FloatArray) -> float:
    return roc_auc_score(y_true=y_true, y_score=y_pred)


def calc_acc(y_true: IntArray,
             y_pred: IntArray) -> float:
    return accuracy_score(y_true=y_true, y_pred=y_pred)
