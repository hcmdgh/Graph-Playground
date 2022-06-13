from .pipeline import * 
dglnn.SAGEConv

if __name__ == '__main__':
    GraphSAGE_Pipeline.run(
        homo_graph_path = '/home/Dataset/GengHao/HomoGraph/DGL/ogbn-arxiv.pt',
        use_gpu = True,
        batch_norm = True,
        use_sampler = True,
        batch_size = 1024,
    )
