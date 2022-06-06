from .imports import * 
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score, adjusted_rand_score, f1_score
from sklearn.linear_model import LogisticRegression

__all__ = ['perform_clustering', 'perform_classification']


def perform_clustering(emb: FloatArray, 
                       label: IntArray, 
                       num_classes: int) -> tuple[float, float]:
    seeds = [random.randint(0, 0x7fffffff) for _ in range(10)]
    nmi_list, ari_list = [], []

    for seed in seeds:
        pred = KMeans(
            n_clusters = num_classes, 
            random_state = seed,
        ).fit_predict(emb)
        
        nmi_list.append(normalized_mutual_info_score(labels_true=label, labels_pred=pred))
        ari_list.append(adjusted_rand_score(labels_true=label, labels_pred=pred))

    nmi = np.mean(nmi_list)
    ari = np.mean(ari_list)

    return nmi, ari 


def perform_classification(emb: FloatArray,
                           label: IntArray,
                           train_mask: BoolArray,
                           eval_mask: BoolArray,
                           max_iter: int = 200,
                           solver: str = 'lbfgs', 
                           multi_class: str = 'auto') -> tuple[float, float]:
    classifier = LogisticRegression(
        solver = solver, 
        multi_class = multi_class,
        max_iter = max_iter,
    )
    
    classifier.fit(X=emb[train_mask], y=label[train_mask]) 

    y_pred = classifier.predict(X=emb[eval_mask])
    y_true = label[eval_mask]
    
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_marco = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    return f1_micro, f1_marco
