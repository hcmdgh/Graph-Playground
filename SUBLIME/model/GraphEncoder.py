from util import * 


class GCNConv(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim)

        self.device = get_device()
        self.to(self.device)

    def forward(self, 
                feat: FloatTensor, 
                adj_mat: FloatTensor) -> FloatTensor:
        hidden = self.fc(feat)
        out = adj_mat @ hidden

        return out


class GraphEncoder(nn.Module):
    @dataclass
    class Param:
        in_dim: int
        hidden_dim_1: int = 512
        hidden_dim_2: int = 256
        out_dim: int = 256
        num_layers: int = 2 
        feat_dropout: float = 0.5
        adj_dropout: float = 0.5
    
    def __init__(self, param: Param):
        super().__init__()
        
        self.device = get_device()
        
        assert param.num_layers >= 2 
        
        self.gnn_layers = nn.ModuleList([
            GCNConv(param.in_dim, param.hidden_dim_1),
            *[
                GCNConv(param.hidden_dim_1, param.hidden_dim_1)
                for _ in range(param.num_layers - 2)
            ],
            GCNConv(param.hidden_dim_1, param.hidden_dim_2),
        ]).to(self.device)
        
        self.proj = nn.Sequential(
            nn.Linear(param.hidden_dim_2, param.out_dim),
            nn.ReLU(),
            nn.Linear(param.out_dim, param.out_dim),
        ).to(self.device)
        
        self.feat_dropout = nn.Dropout(param.feat_dropout)
        self.adj_dropout = nn.Dropout(param.adj_dropout)
        
    def forward(self,
                feat: FloatTensor,
                adj_mat: FloatTensor) -> FloatTensor:
        adj_mat = self.adj_dropout(adj_mat)
        
        h = feat 
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            h = gnn_layer(feat=h, adj_mat=adj_mat)

            if i < len(self.gnn_layers) - 1:
                h = torch.relu(h)
                h = self.feat_dropout(h)
                
        out = self.proj(h)
        
        return out 
