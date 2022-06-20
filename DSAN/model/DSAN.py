from util import * 
from .ResNet import * 
from .LMMD import * 

__all__ = ['DSAN']


class DSAN(nn.Module):
    def __init__(self,
                 in_dim: int, 
                 num_classes: int):
        super().__init__()
        
        self.lmmd = LMMD(num_classes=num_classes)
        
        self.clf_fc = nn.Linear(in_dim, num_classes)
        
        self.device = get_device()
        self.to(self.device)
            
    def forward(self,
                feat_batch_S: FloatTensor,
                feat_batch_T: FloatTensor,
                label_batch_S: IntTensor) -> tuple[FloatTensor, FloatTensor, FloatTensor]:
        # feat_batch_S: float[batch_size x in_dim]
        # feat_batch_T: float[batch_size x in_dim]
        # label_batch_S: int[batch_size]
        
        # pred_S: [batch_size x num_classes]
        pred_S = self.clf_fc(feat_batch_S)
        
        # pred_T: [batch_size x num_classes]
        pred_T = self.clf_fc(feat_batch_T)
        
        lmmd_loss = self.lmmd.calc_loss(
            src_batch = feat_batch_S,
            tgt_batch = feat_batch_T, 
            src_label = label_batch_S,
            tgt_label = torch.softmax(pred_T, dim=1), 
        )
        
        return pred_S, pred_T, lmmd_loss 

    def predict(self, feat_batch: FloatTensor) -> FloatTensor:
        # feat_batch: float[batch_size x in_dim]
        
        # out: [batch_size x num_classes]
        with torch.no_grad():
            out = self.clf_fc(feat_batch)
        
        return out.detach()  
