from .GAT import * 
from util import * 
from config import * 

from dl import * 

__all__ = ['GraphMAE', 'GAT']


class GraphMAE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 emb_dim: int):
        super().__init__()
        
        self.GAT_encoder = GAT(
            in_dim = in_dim, 
            out_dim = emb_dim,
            num_layers = config.GAT_encoder_num_layers,
            num_heads = config.GAT_encoder_num_heads,
            hidden_dim = config.GAT_encoder_hidden_dim, 
        )

        self.GAT_decoder = GAT(
            in_dim = emb_dim, 
            out_dim = in_dim,
            num_layers = config.GAT_decoder_num_layers,
            num_heads = config.GAT_decoder_num_heads,
            hidden_dim = config.GAT_decoder_hidden_dim, 
        )

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
        
        corrupt_feat, corrupt_nids = corrupt_node_feat(
            g = g,
            feat = feat,
            mask_token = self.mask_token,
        )
        
        if config.drop_edge_ratio > 0.:
            raise NotImplementedError
        
        emb = self.encode(g=g, feat=corrupt_feat)
        
        emb = self.fc(emb)
        
        # TODO remask是否是必要的
        emb[corrupt_nids] = 0. 
        
        recon_feat = self.decode(g=g, feat=emb)

        raw_feat = feat[corrupt_nids]
        recon_feat = recon_feat[corrupt_nids]

        loss = calc_SCE_loss(raw_feat, recon_feat, alpha=config.SCE_alpha)
        
        return loss 
