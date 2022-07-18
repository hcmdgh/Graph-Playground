from .Encoder import * 

from dl import * 

__all__ = ['GCA']


class GCA(torch.nn.Module):
    def __init__(self,
                 in_dim: int, 
                 emb_dim: int, 
                 gnn_num_layers: int = 2, 
                 tau: float = 0.5):
        super().__init__()
        
        self.tau = tau
        
        self.encoder = Encoder(
            in_dim = in_dim, 
            emb_dim = emb_dim, 
            gnn_act = F.rrelu,
            gnn_num_layers = gnn_num_layers, 
        )

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        
        self.device = get_device()
        self.to(self.device)

    def forward(self, 
                g: dgl.DGLGraph, 
                feat: FloatTensor) -> FloatTensor:
        out = self.encoder(g=g, feat=feat)

        return out 
    
    # def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
    #     # Space complexity: O(BN) (semi_loss: O(N^2))
    #     device = z1.device
    #     num_nodes = z1.size(0)
    #     num_batches = (num_nodes - 1) // batch_size + 1
    #     f = lambda x: torch.exp(x / self.tau)
    #     indices = torch.arange(0, num_nodes).to(device)
    #     losses = []

    #     for i in range(num_batches):
    #         mask = indices[i * batch_size:(i + 1) * batch_size]
    #         refl_sim = f(calc_cosine_similarity(z1[mask], z1))  # [B, N]
    #         between_sim = f(calc_cosine_similarity(z1[mask], z2))  # [B, N]

    #         losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
    #                                  / (refl_sim.sum(1) + between_sim.sum(1)
    #                                     - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

    #     return torch.cat(losses)
