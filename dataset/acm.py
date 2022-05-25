from util import *

DATASET_PATH = './dataset/bin/ACM/acm.pt'


def load_acm_dataset() -> HeteroGraph:
    return HeteroGraph.load_from_file(DATASET_PATH)
