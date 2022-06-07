from .imports import * 


def split_train_val_test_set(num_total: int,
                             train_ratio: float,
                             val_ratio: float) -> tuple[BoolArray, BoolArray, BoolArray]:
    """
    划分训练集、验证集、测试集，返回mask。
    """
    assert 1. - train_ratio - val_ratio > 0. 
    
    train_mask = np.zeros(num_total, dtype=bool)
    val_mask = np.zeros(num_total, dtype=bool)
    test_mask = np.zeros(num_total, dtype=bool)
    
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)
    
    shuffled_idxs = np.random.permutation(num_total)

    train_idxs = shuffled_idxs[:num_train]
    val_idxs = shuffled_idxs[num_train:num_train+num_val]
    test_idxs = shuffled_idxs[num_train+num_val:]
    
    train_mask[train_idxs] = True 
    val_mask[val_idxs] = True 
    test_mask[test_idxs] = True
    
    assert ~np.all(train_mask & val_mask & test_mask)
    assert np.all(train_mask | val_mask | test_mask)

    return train_mask, val_mask, test_mask 


def sample_negative_edges(positive_edge_set: set[tuple[int, int]],
                          num_nodes: int,
                          sample_cnt: int) -> tuple[IntArray, IntArray]:
    """
    从同构图中随机采样负样本边，要求采样的边不能与正样本边重合，返回格式为(src_index, dest_index)。
    """
    used_edge_set = set(positive_edge_set)
    sampled_edges = [] 

    for _ in range(sample_cnt):
        while True:
            src_idx = random.randrange(num_nodes)
            dest_idx = random.randrange(num_nodes)

            if src_idx != dest_idx \
                    and (src_idx, dest_idx) not in used_edge_set \
                    and (dest_idx, src_idx) not in used_edge_set:
                break 
            
        sampled_edges.append((src_idx, dest_idx))
        used_edge_set.add((src_idx, dest_idx))
        
    src_index, dest_index = np.array(sampled_edges, dtype=np.int64).T 
    
    return src_index, dest_index
