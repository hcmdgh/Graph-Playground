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


class MultiClassificationMetric:
    def __init__(self,
                 status: Literal['train', 'val', 'test']):
        self.status = status 
        self.best_f1_micro_epoch = self.best_f1_macro_epoch = -1 
        self.best_f1_micro = self.best_f1_macro = 0.0 

    def measure(self,
                epoch: int, 
                y_true: IntArray,
                y_pred: IntArray,
                verbose: bool = True) -> tuple[float, float]:
        f1_micro = calc_f1_micro(y_true=y_true, y_pred=y_pred)
        f1_macro = calc_f1_macro(y_true=y_true, y_pred=y_pred)

        if f1_micro > self.best_f1_micro:
            self.best_f1_micro = f1_micro
            self.best_f1_micro_epoch = epoch 
            
        if f1_macro > self.best_f1_macro:
            self.best_f1_macro = f1_macro
            self.best_f1_macro_epoch = epoch 
        
        if verbose:
            logging.info(f"epoch: {epoch}, {self.status}_f1_micro: {f1_micro:.4f}, {self.status}_f1_macro: {f1_macro:.4f}")

            logging.info(f"best_{self.status}_f1_micro: {self.best_f1_micro:.4f} in epoch {self.best_f1_micro_epoch}, "
                        f"best_{self.status}_f1_macro: {self.best_f1_macro:.4f} in epoch {self.best_f1_macro_epoch}")

        return f1_micro, f1_macro 
