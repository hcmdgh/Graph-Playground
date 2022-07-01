from .pipeline import * 
from util import * 

if __name__ == '__main__':
    with open('./MAG/output/g_1.pkl', 'rb') as fp:
        g_1 = pickle.load(fp)
        
    with open('./MAG/output/g_2.pkl', 'rb') as fp:
        g_2 = pickle.load(fp)

    with open('./MAG/output/g_3.pkl', 'rb') as fp:
        g_3 = pickle.load(fp)

    DSAN_pipeline(
        feat_S = g_1.ndata['feat'],
        feat_T = g_2.ndata['feat'],
        label_S = g_1.ndata['label'],
        label_T = g_2.ndata['label'],
    )
