from .imports import * 
from numpy.random import default_rng 


def random_split_dataset(
    *,
    total_cnt: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple[BoolArray, BoolArray, Optional[BoolArray]]:
    if test_ratio is None:
        assert train_ratio + val_ratio == 1. 
    else:
        assert train_ratio + val_ratio + test_ratio == 1. 

    rng = default_rng(seed)
    
    random_idxs = rng.permutation(total_cnt)
    
    if test_ratio is None:
        num_train = int(total_cnt * train_ratio)
        
        train_mask = np.zeros(total_cnt, dtype=bool)
        val_mask = np.zeros(total_cnt, dtype=bool)
    
        train_mask[random_idxs[:num_train]] = True
        val_mask[random_idxs[num_train:]] = True
        
        assert np.all(train_mask | val_mask)
        assert np.all(~(train_mask & val_mask))

        return train_mask, val_mask, None 
    else:
        num_train = int(total_cnt * train_ratio)
        num_val = int(total_cnt * val_ratio)
        
        train_mask = np.zeros(total_cnt, dtype=bool)
        val_mask = np.zeros(total_cnt, dtype=bool)
        test_mask = np.zeros(total_cnt, dtype=bool)
    
        train_mask[random_idxs[:num_train]] = True
        val_mask[random_idxs[num_train: num_train + num_val]] = True
        test_mask[random_idxs[num_train + num_val:]] = True
        
        assert np.all(train_mask | val_mask | test_mask)
        assert np.all(~(train_mask & val_mask & test_mask))

        return train_mask, val_mask, test_mask 
