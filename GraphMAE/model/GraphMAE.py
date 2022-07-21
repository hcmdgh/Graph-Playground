from .GAT import * 
from util import * 
from config import * 

from dl import * 

__all__ = ['GraphMAE', 'GAT']


class GraphMAE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 emb_dim: int,
                 GAT_encoder_param: GAT.Param,
                 GAT_decoder_param: GAT.Param):
        super().__init__()
        
        GAT_encoder_param.in_dim = in_dim
        GAT_encoder_param.out_dim = emb_dim
        self.GAT_encoder = GAT(GAT_encoder_param)

        GAT_decoder_param.in_dim = emb_dim 
        GAT_decoder_param.out_dim = in_dim
        self.GAT_decoder = GAT(GAT_decoder_param)

        self.mask_token = Parameter(torch.zeros(1, in_dim))

        self.fc = nn.Linear(emb_dim, emb_dim)

        self.device = get_device()
        self.to(self.device)
    
    def mask_node_feat(self, feat: FloatTensor) -> tuple[FloatTensor, IntArray]:
        num_nodes = len(feat)
        perm = np.random.permutation(num_nodes)
        
        num_mask_nodes = int(num_nodes * config.mask_ratio)
        mask_nodes = perm[:num_mask_nodes]
        
        out_feat = feat.clone() 
        
        out_feat[mask_nodes] = self.mask_token 
        
        return out_feat, mask_nodes 
    
    def encode(self,
               g: dgl.DGLGraph,
               feat: FloatTensor) -> FloatTensor:
        emb = self.GAT_encoder(g=g, feat=feat)
        
        return emb 
    
    def decode(self,
               g: dgl.DGLGraph,
               feat: FloatTensor) -> FloatTensor:
        emb = self.GAT_decoder(g=g, feat=feat)
        
        return emb         

    def calc_loss(self,
                  g: dgl.DGLGraph,
                  feat: FloatTensor) -> FloatScalarTensor:
        self.train() 
        
        masked_feat, masked_nodes = self.mask_node_feat(feat)
        
        emb = self.encode(g=g, feat=masked_feat)
        
        emb = self.fc(emb)
        
        emb[masked_nodes] = 0. 
        
        recon_feat = self.decode(g=g, feat=emb)

        raw_feat = feat[masked_nodes]
        recon_feat = recon_feat[masked_nodes]

        loss = calc_SCE_loss(raw_feat, recon_feat, alpha=config.SCE_alpha)
        
        return loss 
