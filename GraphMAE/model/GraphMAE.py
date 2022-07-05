from util import * 
from .GAT import * 

__all__ = ['GraphMAE', 'GAT']


class GraphMAE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 emb_dim: int,
                 encoder_gat_param: GAT.Param,
                 decoder_gat_param: GAT.Param,):
        super().__init__()
        
        encoder_gat_param.in_dim = in_dim
        encoder_gat_param.out_dim = emb_dim
        self.encoder_gat = GAT(encoder_gat_param)

        decoder_gat_param.in_dim = emb_dim 
        decoder_gat_param.out_dim = in_dim
        self.decoder_gat = GAT(decoder_gat_param)

        self.device = get_device()
        self.to(self.device)
        
    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        emb = self.encoder_gat(g=g, feat=feat)
        
        return emb  

    def train_graph(self,
                    g: dgl.DGLGraph,
                    feat: FloatTensor) -> FloatScalarTensor:
        self.train() 
        
        emb = self(g=g, feat=feat)
        
        recon_feat = self.decoder_gat(g=g, feat=emb)
        
        loss = F.mse_loss(input=recon_feat, target=feat)
        
        return loss 
    
    def eval_graph(self,
                   g: dgl.DGLGraph,
                   feat: FloatTensor,
                   label: IntTensor,
                   train_mask: BoolTensor,
                   val_mask: BoolTensor,
                   test_mask: BoolTensor) -> dict:
        self.eval() 
        
        with torch.no_grad():
            emb = self(g=g, feat=feat)

        res = mlp_multiclass_classification(
            feat = emb,
            label = label,
            train_mask = train_mask,
            val_mask = val_mask,
            test_mask = test_mask, 
            num_layers = 2, 
            use_tqdm = False, 
        )

        return res 
