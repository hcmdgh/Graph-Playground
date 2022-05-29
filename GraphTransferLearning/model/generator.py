from util import * 
from .graph_sage import * 


class Generator(nn.Module):
    def __init__(self, *,
                 in_dim: int,
                 emb_dim: int,
                 num_layers: int,
                 num_classes: int, 
                 dropout: float = 0.0):
        super().__init__()
        
        hidden_dim = (in_dim + emb_dim) // 2 
        
        self.gnn = GraphSAGE(
            in_dim = in_dim,
            hidden_dim = hidden_dim,
            out_dim = emb_dim,
            num_layers = num_layers,
            aggr_type = 'gcn',
            dropout = dropout,  
        )
        
        self.classifier = MLP(
            in_dim = emb_dim,
            out_dim = num_classes,
            num_layers = 2, 
        )
        
    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        gnn_emb = self.gnn(g, feat)

        clf_out = self.classifier(gnn_emb)
        
        return gnn_emb, clf_out 
