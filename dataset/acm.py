from util import *

DATASET_PATH = './dataset/bin/acm.pt'


def load_acm_dataset() -> HeteroGraph:
    return HeteroGraph.load_from_file(DATASET_PATH)
