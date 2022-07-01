from util import * 
from .GraphEncoder import * 
from .GraphLearner import * 
from ..util import * 

__all__ = ['SUBLIME', 'GraphEncoder']


class SUBLIME(nn.Module):
    def __init__(self,
                 graph_encoder_param: GraphEncoder.Param,
                 feat: FloatTensor,
                 feat_mask_ratio: float = 0.6):
        super().__init__()
        
        self.graph_encoder = GraphEncoder(graph_encoder_param)

        self.graph_learner = FGPLearner(feat=feat)

        self.feat_mask_ratio = feat_mask_ratio
        
    def forward(self,
                feat: FloatTensor,
                anchor_adj_mat: FloatTensor) -> tuple[FloatScalarTensor, FloatTensor]:
        # [BEGIN] Anchor Graph View
        feat_v1 = mask_feat(feat, mask_ratio=self.feat_mask_ratio)
        
        z1 = self.graph_encoder(feat=feat_v1, adj_mat=anchor_adj_mat)
        # [END]
        
        # [BEGIN] Learned Graph View
        feat_v2 = mask_feat(feat, mask_ratio=self.feat_mask_ratio)
        
        learned_adj_mat = self.graph_learner()
        learned_adj_mat = symmetrize_adj_mat(learned_adj_mat)
        learned_adj_mat = normalize_adj_mat(learned_adj_mat, mode='sym')
        
        z2 = self.graph_encoder(feat=feat_v2, adj_mat=learned_adj_mat)
        # [END]
        print(z1.mean(), z2.mean())
        
        loss = self.calc_loss(feat1=z1, feat2=z2)
        
        return loss, learned_adj_mat
        
    @staticmethod
    def calc_loss(feat1: FloatTensor, 
                  feat2: FloatTensor, 
                  temperature: float = 0.2, 
                  symmetric: bool = True,) -> FloatScalarTensor:
        assert len(feat1) == len(feat2)
        
        feat1_norm = torch.norm(feat1, dim=-1)
        feat2_norm = torch.norm(feat2, dim=-1)

        sim_mat = torch.einsum('ik,jk->ij', feat1, feat2) / torch.einsum('i,j->ij', feat1_norm, feat2_norm)
        sim_mat = torch.exp(sim_mat / temperature)
        pos_sim = torch.diagonal(sim_mat)
        
        if symmetric:
            loss_0 = pos_sim / (torch.sum(sim_mat, dim=0) - pos_sim)
            loss_1 = pos_sim / (torch.sum(sim_mat, dim=1) - pos_sim)

            loss_0 = -torch.mean(torch.log(loss_0)) 
            loss_1 = -torch.mean(torch.log(loss_1)) 
            loss = (loss_0 + loss_1) / 2.0

            return loss
        else:
            loss = pos_sim / (torch.sum(sim_mat, dim=1) - pos_sim)
            loss = -torch.mean(torch.log(loss))

            return loss
