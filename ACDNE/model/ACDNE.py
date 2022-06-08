from util import * 
from .NetworkEmbedding import * 
from .NodeClassifier import * 
from .DomainDiscriminator import * 

__all__ = ['ACDNE']


class ACDNE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 num_classes: int,
                 dropout: float = 0.5):
        super().__init__()
        
        hidden_dim = (in_dim + num_classes) // 2 
        
        self.network_embedding = NetworkEmbedding(in_dim=in_dim, out_dim=hidden_dim, dropout=dropout)
        
        self.node_classifier = NodeClassifier(in_dim=hidden_dim, out_dim=num_classes)
        
        self.domain_discriminator = DomainDiscriminator(in_dim=hidden_dim)
        
        self.grl = GradRevLayer()
        
        self.device = get_device() 
        self.to(self.device)
        
    def forward(self,
                feat_self: FloatTensor,
                feat_neigh: FloatTensor) -> tuple[FloatTensor, FloatTensor, FloatTensor]:
        network_emb = self.network_embedding(feat_self=feat_self, feat_neigh=feat_neigh)
        
        pred_logits = self.node_classifier(network_emb)
        
        emb_grl = self.grl(network_emb)
        
        domain_logits = self.domain_discriminator(emb_grl)
        
        return network_emb, pred_logits, domain_logits
