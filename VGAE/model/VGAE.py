from util import * 


class VGAE(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 hidden_dim_1: int, 
                 hidden_dim_2: int):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.feat_encoder = dglnn.GraphConv(in_dim, hidden_dim_1, activation=F.relu, allow_zero_in_degree=True)
        self.mean_encoder = dglnn.GraphConv(hidden_dim_1, hidden_dim_2, activation=None, allow_zero_in_degree=True)
        self.std_encoder = dglnn.GraphConv(hidden_dim_1, hidden_dim_2, activation=None, allow_zero_in_degree=True)

        to_device(self)

    def encoder(self, 
                g: dgl.DGLGraph, 
                feat: FloatTensor) -> dict[str, FloatTensor]:
        # feat: float[num_nodes x emb_dim]
        
        # h: [num_nodes x hidden_dim_1]
        h = self.feat_encoder(g, feat)
        
        # mean/log_std: [num_nodes x hidden_dim_2]
        mean = self.mean_encoder(g, h)
        log_std = self.std_encoder(g, h)
        
        # gaussian_noise/sampled_z: [num_nodes x hidden_dim_2]
        gaussian_noise = to_device(torch.randn(len(feat), self.hidden_dim_2)) 
        sampled_z = mean + gaussian_noise * torch.exp(log_std)

        return {
            'sampled_z': sampled_z,
            'mean': mean, 
            'log_std': log_std, 
        }

    def decoder(self, z: FloatTensor) -> FloatTensor:
        # z: float[num_nodes x hidden_dim_2]
        
        # recon_adj: [num_nodes x num_nodes]
        recon_adj = torch.sigmoid(
            torch.matmul(z, z.t())
        )
        
        return recon_adj

    def forward(self, 
                g: dgl.DGLGraph, 
                feat: FloatTensor) -> dict[str, FloatTensor]:
        # feat: float[num_nodes x emb_dim]
        
        # z: [num_nodes x hidden_dim_2]
        enc_out = self.encoder(g, feat)
        z = enc_out['sampled_z']

        # recon_adj: [num_nodes x num_nodes]
        recon_adj = self.decoder(z)

        out = enc_out
        out['recon_adj'] = recon_adj

        return out 
