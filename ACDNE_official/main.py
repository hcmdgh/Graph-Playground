from .pipeline import * 
from util import * 


def main():
    graph_S = dgl.load_graphs('/home/Dataset/DGL/ogbn_arxiv/g_2018_2018.dgl')[0][0]
    graph_T = dgl.load_graphs('/home/Dataset/DGL/ogbn_arxiv/g_2019_2019.dgl')[0][0]
    
    ACDNE_pipeline(
        graph_S = graph_S,
        graph_T = graph_T, 
    ) 


if __name__ == '__main__':
    main() 
