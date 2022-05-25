from .imports import * 


@dataclass 
class HeteroGraph:
    node_types: set[str]
    num_nodes_dict: dict[str, int]
    edge_index_dict: dict[tuple[str, str, str], tuple[IntTensor, IntTensor]]
    
    # 属性名 -> 结点类型 -> 属性值
    node_prop_dict: dict[str, dict[str, Tensor]]
    
    # 属性名 -> 边类型 -> 属性值
    edge_prop_dict: dict[str, dict[tuple[str, str, str], Tensor]]  
    
    num_classes: Optional[int] = None 
    
    @property
    def edge_types(self) -> set[tuple[str, str, str]]:
        return set(self.edge_index_dict)
    
    def to_dgl(self,
               with_prop: bool = False) -> dgl.DGLHeteroGraph:
        hg = dgl.heterograph(data_dict=self.edge_index_dict,
                             num_nodes_dict=self.num_nodes_dict)
        
        if with_prop:
            for prop_name in self.node_prop_dict:
                for node_type, prop_val in self.node_prop_dict[prop_name].items():
                    hg.nodes[node_type].data[prop_name] = prop_val 
                    
            for prop_name in self.edge_prop_dict:
                for edge_type, prop_val in self.edge_prop_dict[prop_name].items():
                    hg.edges[edge_type].data[prop_name] = prop_val 
                
        return hg 
    
    @classmethod
    def from_dgl(cls, 
                 hg: dgl.DGLHeteroGraph,
                 **kwargs) -> 'HeteroGraph':
        raise NotImplementedError
        return cls(
            node_types = set(hg.ntypes),
            num_nodes_dict = None,
            edge_index_dict = None,
            node_prop_dict = None, 
            edge_prop_dict = None,
        )

    def save_to_file(self, file_path: str):
        torch.save(dataclasses.asdict(self), file_path)
        
    @classmethod
    def load_from_file(cls, file_path: str) -> 'HeteroGraph':
        dict_ = torch.load(file_path)
        
        return cls(**dict_)

    def add_reverse_edges(self):
        rev_edge_index_dict = {}

        for edge_type, edge_index in self.edge_index_dict.items():
            rev_edge_index_dict[(edge_type[2], edge_type[1] + '_rev', edge_type[0])] = (edge_index[1], edge_index[0]) 

        self.edge_index_dict.update(rev_edge_index_dict)


@dataclass 
class HomoGraph:
    num_nodes: int 
    edge_index: tuple[IntTensor, IntTensor]
    node_prop_dict: dict[str, Tensor]
    edge_prop_dict: dict[tuple[str, str, str], Tensor]
    num_classes: Optional[int] = None 
    
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
