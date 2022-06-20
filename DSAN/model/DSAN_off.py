from util import * 
from .ResNet import * 
from .LMMD import * 


class DSAN(nn.Module):
    def __init__(self,
                 num_classes: int,
                 bottle_neck: bool = True):
        super().__init__()
        
        self.resnet = resnet50(pretrained=True)
        
        self.lmmd = LMMD(num_classes=num_classes)
        
        self.bottle_neck = bottle_neck 
        
        if bottle_neck:
            self.bottle_fc = nn.Linear(2048, 256)
            self.clf_fc = nn.Linear(256, num_classes)
        else:
            self.clf_fc = nn.Linear(2048, num_classes)
            
    def forward(self,
                src_img_batch: FloatTensor,
                tgt_img_batch: FloatTensor,
                src_label: IntTensor) -> tuple[FloatTensor, FloatTensor, FloatTensor]:
        # src_img_batch: float[batch_size x C x H x W]
        # tgt_img_batch: float[batch_size x C x H x W]
        # src_label: int[batch_size]
        
        # [BEGIN] Source Prediction
        # src_emb: [batch_size x 2048]
        src_emb = self.resnet(src_img_batch)
        
        if self.bottle_neck:
            # src_emb: [batch_size x emb_dim]
            src_emb = self.bottle_fc(src_emb)
            
        # src_pred: [batch_size x num_classes]
        src_pred = self.clf_fc(src_emb)
        # [END]
        
        # [BEGIN] Target Prediction
        # tgt_emb: [batch_size x 2048]
        tgt_emb = self.resnet(tgt_img_batch)
        
        if self.bottle_neck:
            # tgt_emb: [batch_size x emb_dim]
            tgt_emb = self.bottle_fc(tgt_emb)
            
        # tgt_pred: [batch_size x num_classes]
        tgt_pred = self.clf_fc(tgt_emb)
        # [END]
        
        lmmd_loss = self.lmmd.calc_loss(
            src_batch = src_emb,
            tgt_batch = tgt_emb, 
            src_label = src_label,
            tgt_label = torch.softmax(tgt_pred, dim=1), 
        )
        
        return src_pred, tgt_pred, lmmd_loss 

    def predict(self, img_batch: FloatTensor) -> FloatTensor:
        # img_batch: float[batch_size x C x H x W]
        
        # -> [batch_size x 2048]
        h = self.resnet(img_batch)
        
        if self.bottle_neck:
            # -> [batch_size x emb_dim]
            h = self.bottle_fc(h)
            
        # out: [batch_size x num_classes]
        out = self.clf_fc(h)
        
        return out 
