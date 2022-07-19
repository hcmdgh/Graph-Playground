from .Discriminator import * 

from dl import * 

__all__ = ['MVGRL']


class MVGRL(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 act: Callable = torch.sigmoid):
        super().__init__()
        
        self.gnn_encoder_1 = dglnn.GraphConv(
            in_feats = in_dim,
            out_feats = out_dim,
            norm = 'both',
            activation = nn.PReLU(), 
        )
        
        self.gnn_encoder_2 = dglnn.GraphConv(
            in_feats = in_dim,
            out_feats = out_dim,
            norm = 'none',
            activation = nn.PReLU(), 
        )
        
        self.pooling = dglnn.AvgPooling()
        
        self.discriminator = Discriminator(emb_dim=out_dim)
        
        self.act = act  
        
        self.device = get_device()
        self.to(self.device)
        