from src.argument import parse_args
from src.utils import set_random_seeds
from models import M3S_Trainer
import torch
from dl import * 


def main():
    set_cwd(__file__)

    args = parse_args()
    set_random_seeds(0)
    torch.set_num_threads(2)
    
    embedder = M3S_Trainer(args)
    embedder.train()

if __name__ == "__main__":
    main()