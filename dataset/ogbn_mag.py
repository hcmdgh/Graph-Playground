from util import *

DATASET_PATH = './dataset/bin/ogbn_mag_with_all_feat.pt'


def load_ogbn_mag_dataset() -> HeteroGraph:
    return HeteroGraph.load_from_file(DATASET_PATH)
