from dl import * 

__all__ = ['calc_SCE_loss']


def calc_SCE_loss(v1: FloatTensor,
                  v2: FloatTensor,
                  alpha: float = 1.) -> FloatScalarTensor:
    v1 = F.normalize(v1, p=2, dim=-1)
    v2 = F.normalize(v2, p=2, dim=-1)

    loss = 1 - (v1 * v2).sum(dim=-1)
    loss = loss.pow_(alpha)
    loss = loss.mean() 
    
    return loss 


if __name__ == '__main__':
    a = torch.randn(8, 9)
    bias = torch.randn(8, 9)
    b = a * 5. + bias * 0

    print(calc_SCE_loss(a, b))