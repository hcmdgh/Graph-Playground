from util import * 
from .util import * 


class Projector(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)

        self.reset_parameters() 
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc1.weight, gain)
        nn.init.xavier_normal_(self.fc2.weight, gain)

    def forward(self, inp: FloatTensor) -> FloatTensor:
        return self.fc2(
            F.elu(
                self.fc1(inp)
            )
        )


class ContrastiveLoss(nn.Module):
    def __init__(self,
                 emb_dim: int,
                 tau: float,  # 温度参数
                 lambda_: float):
        super().__init__()
        
        self.tau = tau 
        self.lambda_ = lambda_ 
        
        self.projector = Projector(emb_dim=emb_dim)
        
        self.reset_parameters() 
        
    def reset_parameters(self):
        self.projector.reset_parameters()
        
    # [input]
    #   z_sc: float[num_nodes x emb_dim]
    #   z_mp: float[num_nodes x emb_dim]
    #   positive_mask: bool[num_nodes x num_nodes]
    # [output]
    #   contrastive_loss: float 
    def forward(self,
                z_sc: FloatTensor,
                z_mp: FloatTensor,
                positive_mask: BoolTensor) -> FloatTensor:
        # z_sc: float[num_nodes x emb_dim]
        # z_mp: float[num_nodes x emb_dim]
        # positive_mask: bool[num_nodes x num_nodes]
        
        z_sc_proj = self.projector(z_sc)
        z_mp_proj = self.projector(z_mp)
        
        # sim_sc2mp: [num_nodes x num_nodes]
        sim_sc2mp = torch.exp(calc_cosine_similarity_matrix(z_sc_proj, z_mp_proj) / self.tau)
        sim_mp2sc = sim_sc2mp.T 
        
        # quotient_sc2mp: [num_nodes x num_nodes]
        quotient_sc2mp = sim_sc2mp / (torch.sum(sim_sc2mp, dim=1, keepdim=True) + 1e-8)
        quotient_mp2sc = sim_mp2sc / (torch.sum(sim_mp2sc, dim=1, keepdim=True) + 1e-8)

        positive_mask = positive_mask.to(torch.float32)

        loss_sc = torch.mean(
            -torch.log(
                torch.sum(quotient_sc2mp * positive_mask, dim=1)
            )
        )
        
        loss_mp = torch.mean(
            -torch.log(
                torch.sum(quotient_mp2sc * positive_mask, dim=1)
            )
        )
        
        contrastive_loss = self.lambda_ * loss_sc + (1 - self.lambda_) * loss_mp 
        
        return contrastive_loss 
