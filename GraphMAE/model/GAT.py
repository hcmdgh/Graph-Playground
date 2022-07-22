from config import * 

from dl import * 

__all__ = ['GAT']


class GAT(nn.Module):
    def __init__(self, 
                 *,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 act: Callable = nn.PReLU(),
                 feat_dropout: float = config.GAT_dropout,
                 attn_dropout: float = config.GAT_dropout,
                 negative_slope: float = config.GAT_negative_slope,
                 residual: bool = config.GAT_residual):
        super().__init__()
        
        assert in_dim > 0 and out_dim > 0 
        
        self.act = act  
        
        self.GAT_layers = nn.ModuleList()
        
        if num_layers == 1:
            self.GAT_layers.append(dglnn.GATConv(
                in_feats = in_dim, 
                out_feats = out_dim, 
                num_heads = num_heads,
                feat_drop = feat_dropout,
                attn_drop = attn_dropout, 
                negative_slope = negative_slope,
                residual = residual, 
                activation = None,
            ))
        
        elif num_layers > 1:
            self.GAT_layers.append(dglnn.GATConv(
                in_feats = in_dim, 
                out_feats = hidden_dim, 
                num_heads = num_heads,
                feat_drop = feat_dropout,
                attn_drop = attn_dropout, 
                negative_slope = negative_slope,
                residual = False, 
                activation = self.act,
            ))

            for _ in range(num_layers - 2):
                self.GAT_layers.append(dglnn.GATConv(
                    in_feats = hidden_dim * num_heads, 
                    out_feats = hidden_dim, 
                    num_heads = num_heads,
                    feat_drop = feat_dropout,
                    attn_drop = attn_dropout, 
                    negative_slope = negative_slope,
                    residual = residual, 
                    activation = self.act,
                ))

            self.GAT_layers.append(dglnn.GATConv(
                in_feats = hidden_dim * num_heads, 
                out_feats = out_dim, 
                num_heads = 1,
                feat_drop = feat_dropout,
                attn_drop = attn_dropout, 
                negative_slope = negative_slope,
                residual = residual, 
                activation = None,
            ))
            
        else:
            raise AssertionError 
        
        self.device = get_device()
        self.to(self.device)
        
    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        # feat: [num_nodes x in_dim]
        h = feat 
        
        for l, gat_layer in enumerate(self.GAT_layers):
            # h: [num_nodes x num_heads x out_dim]
            h = gat_layer(g, h)

            if l < len(self.GAT_layers) - 1:
                # h: [num_nodes x (num_heads * out_dim)]
                h = torch.flatten(h, start_dim=1)
            else:
                # h: [num_nodes x out_dim]
                h = torch.mean(h, dim=1)

        # h: [num_nodes x out_dim]
        return h
