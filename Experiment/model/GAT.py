from dl import * 

__all__ = ['GAT']


class GAT(nn.Module):
    @dataclass
    class Param:
        in_dim: int
        out_dim: int
        hidden_dim: int = 8
        num_heads: int = 8
        num_layers: int = 2
        activation: Optional[Callable] = F.elu
        feat_dropout: float = 0.1
        attn_dropout: float = 0.1
        negative_slope: float = 0.2
        residual: bool = False
    
    def __init__(self, param: Param):
        super().__init__()
        
        assert param.in_dim > 0 and param.out_dim > 0 
        
        self.param = param 
        
        self.activation = param.activation 
        
        self.gat_layers = nn.ModuleList()
        
        if param.num_layers == 1:
            self.gat_layers.append(dglnn.GATConv(
                in_feats = param.in_dim, 
                out_feats = param.out_dim, 
                num_heads = param.num_heads,
                feat_drop = param.feat_dropout,
                attn_drop = param.attn_dropout, 
                negative_slope = param.negative_slope,
                residual = param.residual, 
                activation = None,
            ))
        
        elif param.num_layers > 1:
            self.gat_layers.append(dglnn.GATConv(
                in_feats = param.in_dim, 
                out_feats = param.hidden_dim, 
                num_heads = param.num_heads,
                feat_drop = param.feat_dropout,
                attn_drop = param.attn_dropout, 
                negative_slope = param.negative_slope,
                residual = False, 
                activation = self.activation,
            ))

            for _ in range(param.num_layers - 2):
                self.gat_layers.append(dglnn.GATConv(
                    in_feats = param.hidden_dim * param.num_heads, 
                    out_feats = param.hidden_dim, 
                    num_heads = param.num_heads,
                    feat_drop = param.feat_dropout,
                    attn_drop = param.attn_dropout, 
                    negative_slope = param.negative_slope,
                    residual = param.residual, 
                    activation = self.activation,
                ))

            self.gat_layers.append(dglnn.GATConv(
                in_feats = param.hidden_dim * param.num_heads, 
                out_feats = param.out_dim, 
                num_heads = 1,
                feat_drop = param.feat_dropout,
                attn_drop = param.attn_dropout, 
                negative_slope = param.negative_slope,
                residual = param.residual, 
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
        
        for l, gat_layer in enumerate(self.gat_layers):
            # h: [num_nodes x num_heads x out_dim]
            h = gat_layer(g, h)

            if l < len(self.gat_layers) - 1:
                # h: [num_nodes x (num_heads * out_dim)]
                h = torch.flatten(h, start_dim=1)
            else:
                # h: [num_nodes x out_dim]
                h = torch.mean(h, dim=1)

        # h: [num_nodes x out_dim]
        return h

    def train_graph(self,
                    g: dgl.DGLGraph,
                    feat: FloatTensor,
                    mask: BoolTensor, 
                    label: IntTensor) -> FloatScalarTensor:
        self.train() 
        
        logits = self(g=g, feat=feat)
        
        loss = F.cross_entropy(input=logits[mask], target=label[mask])
        
        return loss 
    
    def eval_graph(self,
                   g: dgl.DGLGraph,
                   feat: FloatTensor,
                   mask: BoolTensor, 
                   label: IntTensor) -> tuple[float, float]:
        self.eval() 
        
        with torch.no_grad():
            logits = self(g=g, feat=feat)
            
        y_pred = logits[mask]
        y_true = label[mask]
        
        f1_micro = calc_f1_micro(y_pred=y_pred, y_true=y_true)
        f1_macro = calc_f1_macro(y_pred=y_pred, y_true=y_true)
        
        return f1_micro, f1_macro 
