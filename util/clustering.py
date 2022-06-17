from .imports import * 
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score 

__all__ = ['KMeans_clustering_evaluate', 'KMeans_clustering_predict']


def KMeans_clustering_evaluate(
    feat: FloatArray, 
    num_classes: int,
    label: IntArray, 
    num_runs: int = 10, 
) -> tuple[float, float]:
    seeds = [random.randint(0, 0x7fffffff) for _ in range(num_runs)]
    nmi_list, ari_list = [], []

    for seed in seeds:
        pred = KMeans(
            n_clusters = num_classes, 
            random_state = seed,
        ).fit_predict(feat)
        
        nmi_list.append(normalized_mutual_info_score(labels_true=label, labels_pred=pred))
        ari_list.append(adjusted_rand_score(labels_true=label, labels_pred=pred))

    nmi = np.mean(nmi_list)
    ari = np.mean(ari_list)

    return nmi, ari 


def KMeans_clustering_predict(
    feat: FloatArray, 
    num_classes: int,
) -> FloatArray:
    pred = KMeans(n_clusters=num_classes).fit_predict(feat)

    return pred 
