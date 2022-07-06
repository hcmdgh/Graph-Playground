from dl import * 


class Discriminator(nn.Module):
    def __init__(self,
                 emb_dim: int):
        super().__init__()
        
        self.bfc = nn.Bilinear(emb_dim, emb_dim, 1)
        
    def forward(self,
                h_g_f: FloatTensor,
                h_d_f: FloatTensor,
                h_g_s: FloatTensor,
                h_d_s: FloatTensor,
                p_g_f: FloatTensor,
                p_d_f: FloatTensor) -> FloatTensor:
        p_g_f = p_g_f.expand_as(h_g_f)
        p_d_f = p_d_f.expand_as(h_g_f)

        # positive 
        out_1 = self.bfc(h_d_f, p_g_f).view(-1)
        out_2 = self.bfc(h_g_f, p_d_f).view(-1)

        # negative 
        out_3 = self.bfc(h_d_s, p_g_f).view(-1)
        out_4 = self.bfc(h_g_s, p_d_f).view(-1)
        
        out = torch.cat([out_1, out_2, out_3, out_4], dim=0)
        
        return out 
