from .imports import * 

__all__ = ['GradRevLayer']


# class MLP(nn.Module):
#     def __init__(self,
#                  in_dim: int,
#                  out_dim: int,
#                  num_layers: int = 2):
#         super().__init__()
        
#         assert num_layers > 0 
        
#         dim_arr = np.linspace(in_dim, out_dim, num_layers + 1, dtype=np.int64)
        
#         models = [] 
        
#         for i in range(len(dim_arr) - 2):
#             models.append(nn.Linear(dim_arr[i], dim_arr[i+1]))
#             models.append(nn.ReLU())

#         models.append(nn.Linear(dim_arr[-2], dim_arr[-1]))
            
#         self.seq = nn.Sequential(*models)
        
#     def forward(self, inp: FloatTensor) -> FloatTensor:
#         return self.seq(inp)


class _GradRevLayer(torch.autograd.Function):
    rate = 0.0

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0].neg() * _GradRevLayer.rate
        return grad_output, None


class GradRevLayer(nn.Module):
    def __init__(self, rate: float = 0.5):
        super().__init__()
        
        _GradRevLayer.rate = rate 
    
    def forward(self, inp: FloatTensor) -> FloatTensor:
        out = _GradRevLayer.apply(inp)

        return out 
