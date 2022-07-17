from .GCN import GCN
from .Discriminator import * 
from .Encoder import * 

from dl import * 

__all__ = ['DGI']


class DGI(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 emb_dim: int, 
                 num_gcn_layers: int, 
                 act: Callable, 
                 dropout: float):
        super().__init__()
        
        self.encoder = Encoder(
            in_dim = in_dim, 
            emb_dim = emb_dim, 
            num_layers = num_gcn_layers, 
            act = act, 
            dropout = dropout,
        )
        
        self.discriminator = Discriminator(emb_dim)

        self.device = get_device()
        self.to(self.device)

    def train_graph(self, 
                    g: dgl.DGLGraph, 
                    feat: FloatTensor) -> FloatScalarTensor:
        self.train() 
                    
        positive = self.encoder(g=g, feat=feat, corrupt=False)
        negative = self.encoder(g=g, feat=feat, corrupt=True)
        summary = torch.sigmoid(positive.mean(dim=0))

        positive = self.discriminator(positive, summary)
        negative = self.discriminator(negative, summary)

        l1 = F.binary_cross_entropy_with_logits(input=positive, target=torch.ones_like(positive, device=self.device))
        l2 = F.binary_cross_entropy_with_logits(input=negative, target=torch.zeros_like(negative, device=self.device))

        loss = l1 + l2 
        
        return loss 

    def calc_emb(self,
                 g: dgl.DGLGraph,
                 feat: FloatTensor) -> FloatTensor:
        self.eval() 
        
        with torch.no_grad():
            emb = self.encoder(g=g, feat=feat, corrupt=False)
            emb = emb.detach() 
            
        return emb 
    
    def eval_graph(self,
                   g: dgl.DGLGraph,
                   feat: FloatTensor,
                   label: BoolTensor,
                   train_mask: BoolTensor,
                   val_mask: BoolTensor,
                   test_mask: BoolTensor,
                   num_epochs: int = 200) -> dict[str, float]:
        self.eval() 
        
        emb = self.calc_emb(g=g, feat=feat)
        
        clf_res = mlp_multiclass_classification(
            feat = emb,
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask,
            num_epochs = num_epochs,
        )
        
        return clf_res 
