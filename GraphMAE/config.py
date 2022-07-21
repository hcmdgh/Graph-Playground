from dl import * 


@dataclass
class Config:
    graph: dgl.DGLGraph
    GAT_num_layers: int = 2
    GAT_num_heads: int = 4
    GAT_hidden_dim: int = 128
    GAT_dropout: float = 0.1 
    GAT_negative_slope: float = 0.2 
    GAT_residual: bool = False 
    emb_dim: int = 512
    mask_ratio: float = 0.5
    noise_ratio: float = 0.05
    drop_edge_ratio: float = 0. 
    SCE_alpha: float = 3.
    
    num_epochs: int = 1500 
    lr: float = 0.001
    weight_decay: float = 0.0002
    
    
config = Config(
    graph = load_dgl_dataset('cora'),
)
