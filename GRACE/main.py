from .pipeline import * 

if __name__ == '__main__':
    homo_graph = load_pyg_dataset(
        name = 'Cora', 
        normalize = False,
        add_self_loop = True, 
    )
    
    GRACE_Pipeline.run(
        homo_graph = homo_graph, 
    )
