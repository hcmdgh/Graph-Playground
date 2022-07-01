from .pipeline import * 
from util import * 

if __name__ == '__main__':
    with open('./MAG/output/g_1.pkl', 'rb') as fp:
        g_1 = pickle.load(fp)
        
    with open('./MAG/output/g_2.pkl', 'rb') as fp:
        g_2 = pickle.load(fp)

    with open('./MAG/output/g_3.pkl', 'rb') as fp:
        g_3 = pickle.load(fp)

    GRACE_Pipeline(
        homo_graph = g_2 
    )
