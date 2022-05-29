from util import * 
from torchvision import datasets, transforms

_DATASET_ROOT = './dataset/bin/Office31'


def get_Office31_dataloader(dataset_name: Literal['amazon', 'dslr', 'webcam'],
                            status: Literal['train', 'eval'],
                            batch_size: int) -> DataLoader:
    if status == 'train':
        transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        
        drop_last = True 
        
    elif status == 'eval':
        transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
        
        drop_last = False 
        
    else:
        raise AssertionError 

    dataset = datasets.ImageFolder(root=os.path.join(_DATASET_ROOT, dataset_name), transform=transform)
    
    dataloader = DataLoader(
        dataset = dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        drop_last = drop_last, 
    )
    
    return dataloader 


if __name__ == '__main__':
    l1 = get_Office31_dataloader(dataset_name='amazon', status='train', batch_size=32)
    l2 = get_Office31_dataloader(dataset_name='webcam', status='eval', batch_size=32)
    pass 
