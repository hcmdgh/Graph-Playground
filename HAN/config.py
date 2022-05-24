DEVICE = 'cuda:0'

METAPATHS = [
    [('paper', 'pa', 'author'), ('author', 'ap', 'paper')],
    [('paper', 'pf', 'field'), ('field', 'fp', 'paper')],
]

HIDDEN_DIM = 8 

NUM_LAYERS = 1 

LAYER_NUM_HEADS = 8 

DROPOUT_RATIO = 0.5 

LR = 0.005

WEIGHT_DECAY = 0.001
