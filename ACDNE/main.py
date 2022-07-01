from .pipeline import * 
from util import *

if __name__ == '__main__':
    with open('./MAG/output/g_1.pkl', 'rb') as fp:
        g_1 = pickle.load(fp)
        
    with open('./MAG/output/g_2.pkl', 'rb') as fp:
        g_2 = pickle.load(fp)

    with open('./MAG/output/g_3.pkl', 'rb') as fp:
        g_3 = pickle.load(fp)
        
    print(g_1)
    print(g_2)
    print(g_3)
    # exit()

    ACDNE_pipeline(
        graph_S = g_2,
        graph_T = g_3,
    )
