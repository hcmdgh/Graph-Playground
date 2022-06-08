from .imports import * 


def single_dataloader(idxs: Union[int, IntArray],
                      batch_size: int,
                      shuffle: bool = True,
                      drop_last: bool = False) -> Iterator[IntArray]:
    assert shuffle
                      
    shuffled_idxs = np.random.permutation(idxs)
    N = len(shuffled_idxs)
    
    if N % batch_size == 0:
        batch_cnt = N // batch_size 
    else:
        if drop_last:
            batch_cnt = N // batch_size
        else:
            batch_cnt = N // batch_size + 1 
            
    for i in range(batch_cnt):
        yield shuffled_idxs[i * batch_size: (i + 1) * batch_size]
        
        
def double_dataloader(idxs_1: Union[int, IntArray],
                      idxs_2: Union[int, IntArray],
                      batch_size: int,
                      shuffle: bool = True) -> Iterator[tuple[IntArray, IntArray]]:
    assert shuffle 
                      
    shuffled_idxs_1 = np.random.permutation(idxs_1)
    shuffled_idxs_2 = np.random.permutation(idxs_2)

    max_len = max(len(shuffled_idxs_1), len(shuffled_idxs_2))
    batch_cnt = math.ceil(max_len / batch_size)
    N = batch_cnt * batch_size 
    
    long_1 = shuffled_idxs_1
    long_2 = shuffled_idxs_2
    
    while len(long_1) < N:
        long_1 = np.concatenate([long_1, shuffled_idxs_1])

    while len(long_2) < N:
        long_2 = np.concatenate([long_2, shuffled_idxs_2])

    for i in range(batch_cnt):
        yield long_1[i * batch_size: (i + 1) * batch_size], long_2[i * batch_size: (i + 1) * batch_size]


def combine_dataloaders(dataloader1: DataLoader,
                        dataloader2: DataLoader) -> Iterator[tuple[list[Tensor], list[Tensor]]]:
    len1 = len(dataloader1)
    len2 = len(dataloader2)
    max_step = max(len1, len2) 
    
    cycle1 = itertools.cycle(dataloader1)
    cycle2 = itertools.cycle(dataloader2)

    for step, batch1, batch2 in zip(range(max_step), cycle1, cycle2):
        yield batch1, batch2 
