from util import * 

__all__ = ['calc_loss']


def calc_loss(
    epoch: int, 
    num_epochs: int, 
    label_S: IntTensor,
    pred_S: FloatTensor,
    lmmd_loss: FloatScalarTensor,
    lmmd_weight: float, 
) -> FloatScalarTensor:
    clf_loss = F.cross_entropy(input=pred_S, target=label_S)

    lambda_ = 2 / (1 + math.exp(-10 * epoch / num_epochs)) - 1

    loss = clf_loss + lmmd_weight * lambda_ * lmmd_loss 
    
    return loss 
