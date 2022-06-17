from util import * 


class Discriminator(nn.Module):
    def __init__(self,
                 in_dim: int):
        super().__init__()

        self.bi_fc = nn.Bilinear(in_dim, in_dim, 1)

        self.device = get_device()
        self.to(self.device)

    def forward(self,
                feat_T: FloatTensor,
                feat_F: FloatTensor,
                agg_feat_T: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        """
        [input]
            feat_T: float[num_nodes x in_dim]
            feat_F: float[num_nodes x in_dim]
            agg_feat_T: float[in_dim]
        [output]
            out_T, out_F: float[num_nodes]
        """
        
        agg_feat_T = agg_feat_T.expand_as(feat_T)
        
        # out_T/out_F: [num_nodes]
        out_T = self.bi_fc(feat_T, agg_feat_T).view(-1)
        out_F = self.bi_fc(feat_F, agg_feat_T).view(-1)

        return out_T, out_F
