from .pipeline import * 

if __name__ == '__main__':
    GraphSAGE_Pipeline.run(
        homo_graph_path = '/home/Dataset/GengHao/HomoGraph/DGL/ogbn-arxiv.pt',
        batch_norm = True,
    )
