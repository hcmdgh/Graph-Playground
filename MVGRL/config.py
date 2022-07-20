from dl import * 


@dataclass
class Config:
    graph: dgl.DGLGraph 
    graph_diffusion: Literal['PPR', 'APPNP'] = 'PPR'
    emb_dim: int = 512
    num_epochs: int = 300 
    lr: float = 0.001 
    weight_decay: float = 0.

    # 消融实验
    use_encoder_1_as_emb: bool = False 
    
    
config = Config(
    graph = load_dgl_dataset('cora'),
    # graph = load_ogb_dataset('ogbn-arxiv'),
    graph_diffusion = 'APPNP', 
    num_epochs = 200,
    use_encoder_1_as_emb = False,
)
