from util import * 
from .GCN import * 
from .Discriminator import * 

__all__ = ['DGI']


class DGI(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int):
        super().__init__()
        
        self.gcn = GCN(
            in_dim = in_dim, 
            hidden_dim = 64, 
            out_dim = out_dim,
            num_layers = 1, 
        )
        
        self.discriminator = Discriminator(in_dim=out_dim)

        self.device = get_device()
        
    def forward(self,
                g: dgl.DGLGraph,
                feat_T: FloatTensor,
                feat_F: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        """
        [input]
            feat_T: float[num_nodes x in_dim]
            feat_F: float[num_nodes x in_dim]
        [output]:
            out_T/out_F: float[num_nodes]
        """
                
        # h_T: [num_nodes x out_dim]
        h_T = self.gcn(g=g, feat=feat_T)
        
        # agg_h_T: [out_dim]
        agg_h_T = torch.sigmoid(
            torch.mean(h_T, dim=0) 
        )
        
        # h_F: [num_nodes x out_dim]
        h_F = self.gcn(g=g, feat=feat_F)

        # out_T/out_F: [num_nodes]
        out_T, out_F = self.discriminator(feat_T=h_T, feat_F=h_F, agg_feat_T=agg_h_T)
        
        return out_T, out_F

    def get_embedding(self,
                      g: dgl.DGLGraph,
                      feat: FloatTensor) -> FloatTensor:
        with torch.no_grad():
            h = self.gcn(g=g, feat=feat)

            return h.detach() 
