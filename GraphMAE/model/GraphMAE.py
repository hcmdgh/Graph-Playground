from .GAT import * 
from .loss import * 
from .mask import * 

from dl import * 

__all__ = ['GraphMAE', 'GAT']


class GraphMAE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 emb_dim: int,
                 encoder_gat_param: GAT.Param,
                 decoder_gat_param: GAT.Param,
                 mask_ratio: float = 0.5,
                 SCE_alpha: float = 3.):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.SCE_alpha = SCE_alpha
        
        encoder_gat_param.in_dim = in_dim
        encoder_gat_param.out_dim = emb_dim
        self.encoder_gat = GAT(encoder_gat_param)

        decoder_gat_param.in_dim = emb_dim 
        decoder_gat_param.out_dim = in_dim
        self.decoder_gat = GAT(decoder_gat_param)

        self.mask_token = Parameter(torch.zeros(1, in_dim))

        self.fc = nn.Linear(emb_dim, emb_dim)

        self.device = get_device()
        self.to(self.device)
        
    def train_graph(self,
                    g: dgl.DGLGraph,
                    feat: FloatTensor) -> FloatScalarTensor:
        self.train() 
        
        masked_feat, masked_nodes = mask_node_feat(
            feat = feat,
            mask_token = self.mask_token,
            mask_ratio = self.mask_ratio, 
        )
        
        emb = self.encoder_gat(g=g, feat=masked_feat)
        
        emb = self.fc(emb)
        
        emb[masked_nodes] = 0. 
        
        recon_feat = self.decoder_gat(g=g, feat=emb)

        raw_feat = feat[masked_nodes]
        recon_feat = recon_feat[masked_nodes]

        loss = calc_SCE_loss(raw_feat, recon_feat, alpha=self.SCE_alpha)
        
        return loss 
    
    def eval_graph(self,
                   g: dgl.DGLGraph,
                   feat: FloatTensor,
                   label: IntTensor,
                   train_mask: BoolTensor,
                   val_mask: BoolTensor,
                   test_mask: BoolTensor) -> dict:
        self.eval() 
        
        with torch.no_grad():
            emb = self.encoder_gat(g=g, feat=feat)

        res = sklearn_multiclass_classification(
            feat = emb,
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask, 
            max_epochs = 300, 
        )

        return res 
