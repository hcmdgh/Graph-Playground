from dl import *


def calc_similarity(z1: FloatTensor, z2: FloatTensor) -> FloatTensor:
    """
    [input]
        z1: float[batch_size x emb_dim]
        z1: float[batch_size x emb_dim]
    [output]
        out: float[batch_size x batch_size]
    """
    
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    out = torch.mm(z1, z2.T)

    return out 


def calc_pairwise_loss(z1: FloatTensor, 
                       z2: FloatTensor,
                       tau: float) -> FloatTensor:
    """
    [input]
        z1: float[batch_size x emb_dim]
        z1: float[batch_size x emb_dim]
        tau: float 
    [output]
        loss: [batch_size] 
    """
    
    intra_similarity = torch.exp(calc_similarity(z1, z1) / tau)
    inter_similarity = torch.exp(calc_similarity(z1, z2) / tau)

    loss = -torch.log(
        inter_similarity.diag() / (intra_similarity.sum(dim=-1) + inter_similarity.sum(dim=-1) - intra_similarity.diag())
    )
    
    return loss 
