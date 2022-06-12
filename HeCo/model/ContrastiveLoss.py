from util import * 
from .util import * 


def calc_cosine_similarity_matrix(t1: FloatTensor, t2: FloatTensor) -> FloatTensor:
    # t1: float[B x D]
    # t2: float[B x D]
    
    t1_norm = torch.norm(t1, dim=1, keepdim=True)
    t2_norm = torch.norm(t2, dim=1, keepdim=True)
    numerator = torch.mm(t1, t2.t())
    denominator = torch.mm(t1_norm, t2_norm.t())

    # res: [B x B]，其中res[i, j]表示t1[i]与t2[j]的余弦相似度
    res = numerator / denominator 
    
    return res 


class Projector(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, emb_dim)

        self.reset_parameters() 
        
        self.device = get_device()
        self.to(self.device)
        
    def reset_parameters(self):
        reset_linear_parameters(self.fc1)
        reset_linear_parameters(self.fc2)

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
        
        self.device = get_device()
        
    def reset_parameters(self):
        self.projector.reset_parameters()
        
    # [input]
    #   network_schema_encoder_embedding: float[num_nodes x emb_dim]
    #   metapath_encoder_embedding: float[num_nodes x emb_dim]
    #   positive_sample_mask: bool[num_nodes x num_nodes]
    # [output]
    #   contrastive_loss: float 
    def forward(self,
                network_schema_encoder_embedding: FloatTensor,
                metapath_encoder_embedding: FloatTensor,
                positive_sample_mask: BoolArray) -> FloatScalarTensor:
        # network_schema_encoder_embedding: float[num_nodes x emb_dim]
        # metapath_encoder_embedding: float[num_nodes x emb_dim]
        # positive_sample_mask: bool[num_nodes x num_nodes]
        
        z_sc_proj = self.projector(network_schema_encoder_embedding)
        z_mp_proj = self.projector(metapath_encoder_embedding)
        
        # sim_sc2mp/sim_mp2sc: [num_nodes x num_nodes]
        sim_sc2mp = torch.exp(calc_cosine_similarity_matrix(z_sc_proj, z_mp_proj) / self.tau)
        sim_mp2sc = sim_sc2mp.T 
        
        # quotient_sc2mp: [num_nodes x num_nodes]
        quotient_sc2mp = sim_sc2mp / (torch.sum(sim_sc2mp, dim=1, keepdim=True) + 1e-8)
        quotient_mp2sc = sim_mp2sc / (torch.sum(sim_mp2sc, dim=1, keepdim=True) + 1e-8)

        positive_sample_mask = torch.from_numpy(positive_sample_mask).to(torch.float32).to(self.device)

        loss_sc = torch.mean(
            -torch.log(
                torch.sum(quotient_sc2mp * positive_sample_mask, dim=1)
            )
        )
        
        loss_mp = torch.mean(
            -torch.log(
                torch.sum(quotient_mp2sc * positive_sample_mask, dim=1)
            )
        )
        
        contrastive_loss = self.lambda_ * loss_sc + (1 - self.lambda_) * loss_mp 
        
        return contrastive_loss 
