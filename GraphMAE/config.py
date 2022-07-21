from dl import * 


@dataclass
class Config:
    graph: dgl.DGLGraph
    seed: Optional[int] = None 
    raw_feat_classification: bool = False
    gat_hidden_dim: int = 64
    gat_dropout: float = 0.1 
    emb_dim: int = 64
    mask_ratio: float = 0.5
    SCE_alpha: float = 3.
    
    num_epochs: int = 200 
    lr: float = 0.001
    weight_decay: float = 2e-4
    
    
config = Config(
    graph = load_dgl_dataset('cora'),
)
