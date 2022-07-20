from dl import * 


class Discriminator(nn.Module):
    def __init__(self,
                 emb_dim: int):
        super().__init__()
        
        self.bfc = nn.Bilinear(emb_dim, emb_dim, 1)
        
    def forward(self,
                l_g_f: FloatTensor,
                l_dg_f: FloatTensor,
                l_g_sf: FloatTensor,
                l_dg_sf: FloatTensor,
                g_g_f: FloatTensor,
                g_dg_f: FloatTensor) -> FloatScalarTensor:
        # l_g_f: local - graph - feat
        # l_dg_sf: local - diff_graph - shuffled_feat
        # g_g_f: global - graph - feat 
        
        g_g_f = g_g_f.expand_as(l_g_f)
        g_dg_f = g_dg_f.expand_as(l_g_f)

        # positive 
        pos_pred_1 = self.bfc(l_dg_f, g_g_f).view(-1)
        pos_pred_2 = self.bfc(l_g_f, g_dg_f).view(-1)

        # negative 
        neg_pred_1 = self.bfc(l_dg_sf, g_g_f).view(-1)
        neg_pred_2 = self.bfc(l_g_sf, g_dg_f).view(-1)
        
        y_pred = torch.cat([pos_pred_1, pos_pred_2, neg_pred_1, neg_pred_2], dim=0)
        
        y_true = torch.cat(
            [
                torch.ones_like(pos_pred_1),
                torch.ones_like(pos_pred_2),
                torch.zeros_like(neg_pred_1),
                torch.zeros_like(neg_pred_2),
            ],
            dim = 0,
        )
        
        loss = F.binary_cross_entropy_with_logits(input=y_pred, target=y_true)
        
        return loss 
