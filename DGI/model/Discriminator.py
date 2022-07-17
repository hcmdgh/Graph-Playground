from dl import * 


class Discriminator(nn.Module):
    def __init__(self, 
                 emb_dim: int):
        super().__init__()

        # self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))

        # self.reset_parameters()
        
        self.bc = nn.Bilinear(emb_dim, emb_dim, 1)

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, 
                feat: FloatTensor, 
                summary: FloatTensor) -> FloatTensor:
        # out = torch.matmul(feat, torch.matmul(self.weight, summary))
        summary = summary.expand_as(feat)
        
        out = self.bc(feat, summary)
        out = out.squeeze(dim=-1)

        return out 
