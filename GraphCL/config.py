from dl import * 


@dataclass
class Config:
    graph: dgl.DGLGraph 
    aug_type: str = 'subgraph'
    drop_ratio: float = 0.2 
    
    
config = Config(
    graph = load_dgl_dataset('citeseer'), 
)
