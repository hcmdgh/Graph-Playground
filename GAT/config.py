DEVICE = 'cuda:3'

DATASET = 'ogbn-arxiv'

# [BEGIN] 消融/对比实验
USE_RANDOM_FEAT = False  
# [END]

HIDDEN_DIM = 32

NUM_HEADS = 4

NUM_LAYERS = 2 

DROPOUT_RATIO = 0.0 

USE_RESIDUAL = True   

LR = 0.005

WEIGHT_DECAY = 5e-4 
