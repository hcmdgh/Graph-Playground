from dl import * 
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    'generate_APPNP_graph',
    'generate_PPR_graph',
]


def generate_APPNP_graph(graph: dgl.DGLGraph,
                         k: int = 20,
                         alpha: float = 0.2, 
                         epsilon: float = 0.01) -> tuple[dgl.DGLGraph, FloatTensor]:
    assert is_on_cpu(graph)
                         
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
                         
    appnp = dglnn.APPNPConv(k=k, alpha=alpha)
    feat = torch.eye(graph.num_nodes())

    diff_adj = appnp(graph=graph, feat=feat).numpy()
    diff_adj[diff_adj < epsilon] = 0.

    scaler = MinMaxScaler()
    diff_adj = scaler.fit_transform(diff_adj)

    diff_edge_index = np.nonzero(diff_adj)
    diff_graph = dgl.graph(diff_edge_index)
    diff_edge_weight = diff_adj[diff_edge_index]
    diff_edge_weight = torch.from_numpy(diff_edge_weight).to(torch.float32)

    return diff_graph, diff_edge_weight


def generate_PPR_graph(graph: dgl.DGLGraph, 
                       alpha: float = 0.2) -> tuple[dgl.DGLGraph, FloatTensor]:
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
             
    N = graph.num_nodes() 
    A = graph.adj().to_dense().cpu().numpy() 
    
    D = np.sum(A, axis=-1)
    D_ = np.power(D, -0.5)
    D_ = np.diag(D_)
    
    A_ = D_ @ A @ D_ 
    
    PPR_mat = alpha * np.linalg.inv((np.eye(N) - (1 - alpha) * A_))
    assert PPR_mat.shape == (N, N) 
    
    diff_edge_index = np.nonzero(PPR_mat)
    diff_graph = dgl.graph(diff_edge_index)
    diff_edge_weight = PPR_mat[diff_edge_index]
    diff_edge_weight = torch.tensor(diff_edge_weight, dtype=torch.float32)
    
    return diff_graph, diff_edge_weight
