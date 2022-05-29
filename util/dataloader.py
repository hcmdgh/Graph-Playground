from .imports import * 


def combine_dataloaders(dataloader1: DataLoader,
                        dataloader2: DataLoader) -> Iterator[tuple[list[Tensor], list[Tensor]]]:
    len1 = len(dataloader1)
    len2 = len(dataloader2)
    max_step = max(len1, len2) 
    
    cycle1 = itertools.cycle(dataloader1)
    cycle2 = itertools.cycle(dataloader2)

    for step, batch1, batch2 in zip(range(max_step), cycle1, cycle2):
        yield batch1, batch2 
