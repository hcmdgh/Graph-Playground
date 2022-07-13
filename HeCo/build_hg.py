from utils import load_data

from dl import * 

if __name__ == '__main__':
    set_cwd(__file__)
    
    nei_index, feats, mps, pos, label, idx_train, idx_val, idx_test = \
        load_data('acm', [20, 40, 60], [4019, 7167, 60])
        
    dataset_dict = {
        'feat': {
            ''
        },
    }