DEVICE = 'cuda:2'

DATASET = 'acm'

# [BEGIN] 消融/对比实验
REMOVE_SOFTMAX_IN_SEMANTIC_ATTN = False  

REMOVE_SEMANTIC_ATTN = True 
# [END]

if DATASET == 'acm':
    NUM_EPOCHS = 500 
    
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

elif DATASET == 'ogbn_mag':
    METAPATHS = [
        [('paper', 'rev_writes', 'author'), ('author', 'writes', 'paper')],
        [('paper', 'has_topic', 'field_of_study'), ('field_of_study', 'rev_has_topic', 'paper')],
    ]

    HIDDEN_DIM = 8 

    NUM_LAYERS = 1 

    LAYER_NUM_HEADS = 8 

    DROPOUT_RATIO = 0.5 

    LR = 0.005

    WEIGHT_DECAY = 0.001
    
else:
    raise AssertionError 
