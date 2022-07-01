from util import * 
from ..util import * 


# TODO 目前为随机初始化
class FGPLearner(nn.Module):
    def __init__(self,
                 feat: FloatTensor):
        super().__init__()

        num_nodes = len(feat)
        
        # self.adj = Parameter(
        #     torch.zeros(num_nodes, num_nodes).uniform_(-1, 0)
        # )
        
        self.adj = Parameter(
            torch.from_numpy(nearest_neighbors_pre_elu(feat.cpu().numpy(), 30, 'cosine', 6))
        )
        
        self.device = get_device()
        self.to(self.device)
        
    def forward(self) -> FloatTensor:
        out = F.elu(self.adj) + 1.
        
        return out 
