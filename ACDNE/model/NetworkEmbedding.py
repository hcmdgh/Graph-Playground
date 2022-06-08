from util import * 
from .FeatureExtractor import * 


class NetworkEmbedding(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 dropout: float = 0.5):
        super().__init__()
        
        hidden_dim = (in_dim + out_dim) // 2
        
        self.feature_extractor_self = FeatureExtractor(in_dim=in_dim, out_dim=hidden_dim)
        self.feature_extractor_neigh = FeatureExtractor(in_dim=in_dim, out_dim=hidden_dim)

        self.fc = nn.Linear(hidden_dim * 2, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                feat_self: FloatTensor,
                feat_neigh: FloatTensor) -> FloatTensor:
        h_self = self.feature_extractor_self(feat_self)
        h_neigh = self.feature_extractor_neigh(feat_neigh)

        h = torch.cat([h_self, h_neigh], dim=-1)
        
        out = torch.relu(
            self.fc(h)
        )
        
        return out 
