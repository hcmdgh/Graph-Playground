from util import * 

__all__ = ['HeteroGraph']


@dataclass
class Edge:
    src_nid: int 
    dest_nid: int 
    year: int 


class HeteroGraph:
    def __init__(self):
        self.adj_list_dict: dict[EdgeType, dict[int, list[Edge]]] = dict()
        self.node_feat_dict: dict[str, dict[NodeType, FloatTensor]] = dict() 