from util import * 

DATASET_ROOT = './dataset/bin/ACDNE'


def load_ACDNE_dataset(dataset_name: str) -> HomoGraph:
    if dataset_name == 'acmv9':
        return load_acmv9_dataset() 
    elif dataset_name == 'dblpv7':
        return load_dblpv7_dataset()
    elif dataset_name == 'citationv1':
        return load_citationv1_dataset()
    else:
        raise AssertionError


def load_acmv9_dataset() -> HomoGraph:
    return HomoGraph.load_from_file(os.path.join(DATASET_ROOT, 'acmv9.pt'))


def load_dblpv7_dataset() -> HomoGraph:
    return HomoGraph.load_from_file(os.path.join(DATASET_ROOT, 'dblpv7.pt'))


def load_citationv1_dataset() -> HomoGraph:
    return HomoGraph.load_from_file(os.path.join(DATASET_ROOT, 'citationv1.pt'))
