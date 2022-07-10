from dl import * 
from sklearn.preprocessing import normalize, StandardScaler
from torch_geometric.data import Data, Batch

    
class PersonalizedPageRank:
    def __init__(self, adj_mat, maxsize=200, n_order=2):
        self.n_order = n_order
        self.maxsize = maxsize
        self.adj_mat = adj_mat
        self.P = normalize(adj_mat, norm='l1', axis=0)
        self.d = np.array(adj_mat.sum(1)).squeeze()
        
    def search(self, seed, alpha=0.85):
        x = sp.csc_matrix((np.ones(1), ([seed], np.zeros(1, dtype=int))), shape=[self.P.shape[0], 1])
        r = x.copy()
        for _ in range(self.n_order):
            x = (1 - alpha) * r + alpha * self.P @ x
        scores = x.data / (self.d[x.indices] + 1e-9)
        
        idx = scores.argsort()[::-1][:self.maxsize]
        neighbor = np.array(x.indices[idx])
        
        seed_idx = np.where(neighbor == seed)[0]
        if seed_idx.size == 0:
            neighbor = np.append(np.array([seed]), neighbor)
        else :
            seed_idx = seed_idx[0]
            neighbor[seed_idx], neighbor[0] = neighbor[0], neighbor[seed_idx]
            
        assert np.where(neighbor == seed)[0].size == 1
        assert np.where(neighbor == seed)[0][0] == 0
        
        return neighbor
    
    def search_all(self, 
                   num_nodes: int) -> dict[int, IntArray]:
        nid_2_neighbors = dict() 
            
        for nid in tqdm(range(num_nodes), desc='Sampling neighbors'):
            nid_2_neighbors[nid] = self.search(nid)

        return nid_2_neighbors
    
    
class SubgraphService:
    def __init__(self,
                 graph: dgl.DGLGraph,
                 neighbor_sample_cnt: int = 20,
                 PPR_order: int = 10):
        edge_index = graph.edges()
        edge_index = (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())
        
        self.num_nodes = graph.num_nodes() 
        self.num_edges = len(edge_index[0])

        sp_adj_mat = sp.csr_matrix(
            (np.ones(self.num_edges, dtype=np.float32), edge_index),
            shape = [self.num_edges, self.num_edges],
        )
        
        self.ppr = PersonalizedPageRank(adj_mat=sp_adj_mat, n_order=PPR_order)
        
        self.nid_2_subgraph = dict()
        
        self.nid_2_neighbors = self.ppr.search_all(self.num_nodes)

        for nid in tqdm(range(self.num_nodes), desc='Extracting subgraphs'):
            neighbors = self.nid_2_neighbors[nid][:neighbor_sample_cnt]
            assert neighbors[0] == nid 
            
            subgraph = dgl.node_subgraph(graph=graph, nodes=neighbors)
            subgraph = dgl.remove_self_loop(subgraph)
            subgraph = dgl.add_self_loop(subgraph)
            
            self.nid_2_subgraph[nid] = subgraph

    def extract_subgraph_batch(self, nids: Iterable[int]) -> tuple[dgl.DGLGraph, IntArray]:
        subgraph_list = []
        center_nid_list = [] 
        center_nid = 0 

        for nid in nids:
            subgraph = self.nid_2_subgraph[nid]
            subgraph_list.append(subgraph)
            
            center_nid_list.append(center_nid)
            center_nid += subgraph.num_nodes() 
        
        subgraph_batch = dgl.batch(subgraph_list)
        center_nids = np.array(center_nid_list, dtype=np.int64)
        
        return subgraph_batch, center_nids

    def save_to_file(self, path: str):
        with open(path, 'wb') as fp:
            pickle.dump(self, fp)

    @classmethod 
    def load_from_file(cls, path: str) -> 'SubgraphService':
        with open(path, 'rb') as fp:
            return pickle.load(fp)
