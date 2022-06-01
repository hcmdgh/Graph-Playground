from util import * 
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, normalized_mutual_info_score, adjusted_rand_score


def calc_cosine_similarity_matrix(t1: FloatTensor, t2: FloatTensor) -> FloatTensor:
    # t1: float[B x D]
    # t2: float[B x D]
    
    t1_norm = torch.norm(t1, dim=1, keepdim=True)
    t2_norm = torch.norm(t2, dim=1, keepdim=True)
    numerator = torch.mm(t1, t2.t())
    denominator = torch.mm(t1_norm, t2_norm.t())

    # res: [B x B]，其中res[i, j]表示t1[i]与t2[j]的余弦相似度
    res = numerator / denominator 
    
    return res 


def node_clustering(emb: FloatArray, 
                    labels: IntArray, 
                    num_classes: int) -> tuple[float, float]:
    seeds = [random.randint(0, 0x7fffffff) for _ in range(10)]
    nmi_list, ari_list = [], []

    for seed in seeds:
        pred = KMeans(
            n_clusters = num_classes, 
            random_state = seed,
        ).fit_predict(emb)
        
        nmi_list.append(normalized_mutual_info_score(labels_true=labels, labels_pred=pred))
        ari_list.append(adjusted_rand_score(labels_true=labels, labels_pred=pred))

    nmi = np.mean(nmi_list)
    ari = np.mean(ari_list)

    return nmi, ari 


def node_classification(emb: FloatTensor,
                        labels: IntTensor,
                        train_mask: Union[BoolTensor, BoolArray],
                        eval_mask: Union[BoolTensor, BoolArray],
                        verbose: bool = True) -> tuple[float, float]:
    emb = to_device(emb)
    labels = to_device(labels)
                        
    num_classes = len(torch.unique(labels))
    assert int(torch.min(labels)) == 0 and int(torch.max(labels)) == num_classes - 1 

    emb_dim = emb.shape[-1]
    
    clf_model = nn.Sequential(
        nn.Linear(emb_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )
    clf_model = to_device(clf_model)
    
    optimizer = optim.Adam(clf_model.parameters(), lr=0.01)
    
    early_stopping = EarlyStopping(monitor_epochs=50)

    metric = MultiClassificationMetric(status='val')
    
    for epoch in itertools.count(1):
        # [BEGIN] Train 
        clf_model.train() 
        
        logits = clf_model(emb)
        
        loss = F.cross_entropy(input=logits[train_mask], target=labels[train_mask])
        eval_loss = F.cross_entropy(input=logits[eval_mask], target=labels[eval_mask])
        early_stopping.record_loss(eval_loss)
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
        if verbose:
            logging.info(f"epoch: {epoch}, train_loss: {float(loss):.4f}, eval_loss: {float(eval_loss):.4f}")
        # [END]
        
        # [BEGIN] Eval 
        clf_model.eval() 
        
        y_pred = logits[eval_mask].detach().cpu().numpy() 
        y_pred = np.argmax(y_pred, axis=-1)
            
        y_true = labels[eval_mask].cpu().numpy() 
            
        metric.measure(epoch=epoch, y_true=y_true, y_pred=y_pred, verbose=verbose)

        if early_stopping.should_stop:
            break 
        # [END]
        
    best_f1_micro = metric.best_f1_micro
    best_f1_macro = metric.best_f1_macro
         
    return best_f1_micro, best_f1_macro
