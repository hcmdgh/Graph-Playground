from util import * 
from torchvision import datasets
import torchvision.transforms as transforms

_DATASET_ROOT = './dataset/bin/MNIST'


def load_MNIST_dataset() -> Dataset:
    return datasets.MNIST(
        root = _DATASET_ROOT,
        train = True,
        download = False,
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5]),
        ]),
    )
