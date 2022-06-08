from .imports import * 


class EdgeIndex:
    def __init__(self, 
                 edge_index: tuple[IntTensor, IntTensor]):
        src_index, dest_index = edge_index 
        src_index = src_index.cpu().numpy()
        dest_index = dest_index.cpu().numpy()
        assert len(src_index) == len(dest_index)
        
        self.adj_list: dict[int, list[int]] = defaultdict(list)
                 
        for src_nid, dest_nid in zip(src_index, dest_index):
            src_nid, dest_nid = int(src_nid), int(dest_nid) 

            self.adj_list[src_nid].append(dest_nid)
            
    def sample_neighbors(self,
                         nid_batch: IntTensor,
                         num_neighbors: int) -> IntTensor:
        device = nid_batch.device 
                         
        neighbor_list = [] 
        
        for nid in nid_batch:
            nid = int(nid)
            
            neighbor_nids = self.adj_list[nid]
            
            if not neighbor_nids:
                neighbor_list.append(
                    torch.full(size=[num_neighbors], 
                               fill_value=-1, 
                               dtype=torch.int64,
                               device=device)
                )
            else:
                neighbor_list.append(
                    torch.tensor(random.choices(neighbor_nids, k=num_neighbors),
                                 dtype=torch.int64,
                                 device=device)
                )
            
        neighbor_th = torch.stack(neighbor_list)
        assert neighbor_th.shape == (len(nid_batch), num_neighbors)
        
        return neighbor_th 
    

class EdgeIndex_高效版:
    def __init__(self, 
                 edge_index: tuple[IntTensor, IntTensor],
                 num_nodes_S: Optional[int] = None,
                 num_nodes_T: Optional[int] = None,
                 first_nid_S: int = 0,
                 first_nid_T: int = 0):
        self.device = edge_index[0].device 
        
        src_index, dest_index = edge_index 
        assert len(src_index) == len(dest_index) > 0 
        
        # [BEGIN] 按结点下标排序
        sorted_index = torch.argsort(src_index)
        
        self.src_index = src_index[sorted_index]
        self.dest_index = dest_index[sorted_index]
        # [END]
        
        if not num_nodes_S:
            num_nodes_S = int(torch.max(self.src_index)) - first_nid_S + 1 
        else:
            assert num_nodes_S >= int(torch.max(self.src_index)) - first_nid_S + 1 
        if not num_nodes_T:
            num_nodes_T = int(torch.max(self.dest_index)) - first_nid_T + 1 
        else:
            assert num_nodes_T >= int(torch.max(self.dest_index)) - first_nid_T + 1 

        self.num_nodes_S = num_nodes_S
        self.num_nodes_T = num_nodes_T
        self.first_nid_S = first_nid_S 
        self.first_nid_T = first_nid_T 
        
        # 结点下标统一为从0开始
        self.src_index -= self.first_nid_S
        self.dest_index -= self.first_nid_T 
        
        # dest_index末尾补-1
        self._dest_index = torch.cat([self.dest_index, torch.tensor([-1], device=self.device)])
        
        # [BEGIN] 构建每一个源点对应的边下标范围
        # 对于每一个源点，其对应的边下标范围[begin_idx, end_idx)
        self.eid_bound_map = torch.full(size=[self.num_nodes_S, 2], fill_value=len(self.src_index), dtype=torch.int64, device=self.device)
        
        last_nid = -1 

        for eid, nid in enumerate(self.src_index):
            nid = int(nid)
            
            if nid != last_nid:
                self.eid_bound_map[nid, 0] = eid 
                
                if last_nid > -1:
                    self.eid_bound_map[last_nid, 1] = eid
                
            last_nid = nid     
             
        if last_nid > -1:
            self.eid_bound_map[last_nid, 1] = len(self.src_index)
        # [END]
        
    def sample_neighbors(self,
                         nid_batch: IntTensor,
                         num_neighbors: int) -> IntTensor:
        batch_size = len(nid_batch)
                         
        # rand_mat: float[batch_size x num_neighbors]
        rand_mat = torch.rand(batch_size, num_neighbors, device=self.device)
        
        # num_neighbors_arr: int[batch_size x 1]
        num_neighbors_arr = (self.eid_bound_map[nid_batch, 1] - self.eid_bound_map[nid_batch, 0]).view(-1, 1)

        # neighbor_eid_mat: int[batch_size x num_neighbors]
        neighbor_eid_mat = self.eid_bound_map[nid_batch, 0].view(-1, 1) + (num_neighbors_arr * rand_mat).to(torch.int64)

        # neighbor_idx_mat: int[batch_size x num_neighbors]
        neighbor_idx_mat = self._dest_index[neighbor_eid_mat]

        # 结点下标增加偏移量
        neighbor_idx_mat += self.first_nid_T 

        return neighbor_idx_mat 


class HeteroGraph:
    def __init__(self, 
                 *,
                 node_types: Optional[set[str]] = None,
                 edge_types: Optional[set[tuple[str, str, str]]] = None,
                 num_nodes_dict: Optional[dict[str, int]] = None,
                 num_edges_dict: Optional[dict[tuple[str, str, str], int]] = None,
                 edge_index_dict: dict[tuple[str, str, str], tuple[IntTensor, IntTensor]],
                 node_attr_dict: Optional[dict[str, dict[str, Tensor]]] = None,
                 edge_attr_dict: Optional[dict[str, dict[tuple[str, str, str], Tensor]]] = None,
                 num_classes: Optional[int] = None):
        for edge_type, (src_edge_index, dest_edge_index) in edge_index_dict.items():
            if isinstance(src_edge_index, IntTensor):
                pass 
            elif isinstance(src_edge_index, IntArray):
                edge_index_dict[edge_type] = (torch.from_numpy(src_edge_index), torch.from_numpy(dest_edge_index))    
            elif isinstance(src_edge_index, list):
                edge_index_dict[edge_type] = (
                    torch.tensor(src_edge_index, dtype=torch.int64),
                    torch.tensor(dest_edge_index, dtype=torch.int64),
                )
            else:
                raise AssertionError
    
        self.edge_index_dict = edge_index_dict 
        self.num_classes = num_classes
        
        if not node_types:
            self.node_types = set() 
            
            for src_type, _, dest_type in edge_index_dict:
                self.node_types.add(src_type) 
                self.node_types.add(dest_type) 
        else:
            self.node_types = node_types 
            
        if not edge_types:
            self.edge_types = set(edge_index_dict)
        else:
            self.edge_types = edge_types 

        if not num_nodes_dict:
            self.num_nodes_dict = dict()

            for (src_type, _, dest_type), (src_edge_index, dest_edge_index) in edge_index_dict.items():
                n = int(torch.max(src_edge_index)) + 1 
                self.num_nodes_dict[src_type] = max(n, self.num_nodes_dict.get(src_type, n))
                
                n = int(torch.max(dest_edge_index)) + 1 
                self.num_nodes_dict[dest_type] = max(n, self.num_nodes_dict.get(dest_type, n))
        else:
            self.num_nodes_dict = num_nodes_dict 

        assert not num_edges_dict 
        self.num_edges_dict = {edge_type: len(src_edge_index) for edge_type, (src_edge_index, dest_edge_index) in edge_index_dict.items()}
            
        if not node_attr_dict:
            self.node_attr_dict = dict() 
        else:
            for attr_name in node_attr_dict:
                for node_type, attr_val in node_attr_dict[attr_name].items():
                    assert len(attr_val) == self.num_nodes_dict[node_type] 
            
            self.node_attr_dict = node_attr_dict

        if not edge_attr_dict:
            self.edge_attr_dict = dict() 
        else:
            for attr_name in edge_attr_dict:
                for edge_type, attr_val in edge_attr_dict[attr_name].items():
                    assert len(attr_val) == self.num_edges_dict[edge_type] 
            
            self.edge_attr_dict = edge_attr_dict
    
    def to_dgl(self,
               with_attr: bool = False) -> dgl.DGLHeteroGraph:
        hg = dgl.heterograph(data_dict=self.edge_index_dict,
                             num_nodes_dict=self.num_nodes_dict)
        
        if with_attr:
            for attr_name in self.node_attr_dict:
                for node_type, attr_val in self.node_attr_dict[attr_name].items():
                    hg.nodes[node_type].data[attr_name] = attr_val 
                    
            for attr_name in self.edge_attr_dict:
                for edge_type, attr_val in self.edge_attr_dict[attr_name].items():
                    hg.edges[edge_type].data[attr_name] = attr_val 
                
        return hg 
    
    @classmethod
    def from_dgl(cls, 
                 hg: dgl.DGLHeteroGraph) -> 'HeteroGraph':
        raise NotImplementedError

    def save_to_file(self, file_path: str):
        torch.save(self.__dict__, file_path)
        
    @classmethod
    def load_from_file(cls, file_path: str) -> 'HeteroGraph':
        dict_ = torch.load(file_path)
        
        return cls(**dict_)


class HomoGraph:
    def __init__(self,
                 *,
                 num_nodes: Optional[int] = None,  
                 num_edges: Optional[int] = None,  
                 edge_index: tuple[IntTensor, IntTensor],
                 node_attr_dict: Optional[dict[str, Tensor]] = None,
                 edge_attr_dict: Optional[dict[tuple[str, str, str], Tensor]] = None,
                 num_classes: Optional[int] = None) -> None:
        src_edge_index, dest_edge_index = edge_index
                 
        if isinstance(src_edge_index, IntTensor):
            pass 
        elif isinstance(src_edge_index, IntArray):
            src_edge_index = torch.from_numpy(src_edge_index)
            dest_edge_index = torch.from_numpy(dest_edge_index)

            edge_index = (src_edge_index, dest_edge_index)    
        elif isinstance(src_edge_index, list):
            src_edge_index = torch.tensor(src_edge_index, dtype=torch.int64)
            dest_edge_index = torch.tensor(dest_edge_index, dtype=torch.int64)

            edge_index = (src_edge_index, dest_edge_index)    
        else:
            raise AssertionError

        if not num_nodes:
            self.num_nodes = max(
                int(torch.max(src_edge_index)) + 1,
                int(torch.max(dest_edge_index)) + 1,
            )
        else:
            self.num_nodes = num_nodes 
            
        assert not num_edges 
        self.num_edges = len(src_edge_index)
        
        if not node_attr_dict:
            self.node_attr_dict = dict() 
        else:
            for attr_name, attr_val in node_attr_dict.items():
                assert len(attr_val) == self.num_nodes
            
            self.node_attr_dict = node_attr_dict
            
        if not edge_attr_dict:
            self.edge_attr_dict = dict() 
        else:
            for attr_name, attr_val in edge_attr_dict.items():
                assert len(attr_val) == self.num_edges
            
            self.edge_attr_dict = edge_attr_dict
    
    
    def to_dgl(self,
               with_prop: bool = False) -> dgl.DGLGraph:
        g = dgl.graph(data=self.edge_index,
                       num_nodes=self.num_nodes)
        
        if with_prop:
            for prop_name, prop_val in self.node_prop_dict.items():
                g.ndata[prop_name] = prop_val 
                    
            for prop_name, prop_val in self.edge_prop_dict.items():
                g.edata[prop_name] = prop_val 

        return g 
    
    @classmethod
    def from_dgl(cls, g: dgl.DGLGraph) -> 'HomoGraph':
        return cls(
            num_nodes = g.num_nodes(),
            edge_index = tuple(g.edges()),
            node_prop_dict = dict(g.ndata),
            edge_prop_dict = dict(g.edata),
        )

    def save_to_file(self, file_path: str):
        torch.save(dataclasses.asdict(self), file_path)
        
    @classmethod
    def load_from_file(cls, file_path: str) -> 'HomoGraph':
        dict_ = torch.load(file_path)
        
        return cls(**dict_)
