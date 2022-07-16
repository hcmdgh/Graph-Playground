from dl import * 


class GraphSAGE(nn.Module):
    def __init__(self,
                 *, 
                 in_dim: int,
                 hidden_dim: int = 16,
                 out_dim: int,
                 num_layers: int = 2,
                 act: Callable = torch.relu,
                 dropout: float = 0.1,
                 aggr_type: str = 'gcn'):
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.act = act
        assert num_layers >= 2 

        self.layers.append(dglnn.SAGEConv(in_feats=in_dim, out_feats=hidden_dim, aggregator_type=aggr_type))

        for _ in range(num_layers - 2):
            self.layers.append(dglnn.SAGEConv(in_feats=hidden_dim, out_feats=hidden_dim, aggregator_type=aggr_type))

        self.layers.append(dglnn.SAGEConv(in_feats=hidden_dim, out_feats=out_dim, aggregator_type=aggr_type))

        self.device = get_device()
        self.to(self.device)

    def forward(self, 
                g: dgl.DGLGraph, 
                feat: FloatTensor) -> FloatTensor:
        h = self.dropout(feat)
        
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            
            if l < len(self.layers) - 1:
                h = self.act(h)
                h = self.dropout(h)

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
