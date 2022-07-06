from dl import * 


def calc_PPR_mat(graph: dgl.DGLGraph, 
                 alpha: float = 0.2, 
                 add_self_loop: bool = True) -> FloatArray:
    if add_self_loop:
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
             
    N = graph.num_nodes() 
             
    A = graph.adj().to_dense().cpu().numpy() 
    
    D = np.sum(A, axis=-1)
    D_ = np.power(D, -0.5)
    D_ = np.diag(D_)
    
    A_ = D_ @ A @ D_ 
    
    out = alpha * np.linalg.inv((np.eye(N) - (1 - alpha) * A_)) 
    
    return out 


if __name__ == '__main__':
    graph = load_dgl_dataset('cora')
    
    calc_PPR_mat(graph)