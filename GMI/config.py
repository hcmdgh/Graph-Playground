from dl import *  


@dataclass 
class Config:
    graph: dgl.DGLGraph 
    hidden_dim: int = 512
    num_negative_samples: int = 5 
    alpha: float = 0.8 
    beta: float = 1.0 
    gamma: float = 1.0 
    gcn_act: Callable = nn.PReLU()
    
    num_epochs: int = 550 
    lr: float = 0.001 
    weight_decay: float = 0.0 
    
    @classmethod 
    def cora_config(cls) -> 'Config':
        return cls(
            graph = load_dgl_dataset('cora'), 
        ) 
