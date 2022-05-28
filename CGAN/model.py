from util import * 


class Generator(nn.Module):
    def __init__(self,
                 num_classes: int,
                 latent_dim: int,
                 img_shape: list[int]):
        super().__init__()
        
        self.img_shape = img_shape 

        self.label_emb = nn.Embedding(num_embeddings=num_classes,
                                      embedding_dim=num_classes)

        self.seq = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, eps=0.8), 
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, eps=0.8), 
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, eps=0.8), 
            nn.LeakyReLU(0.2),
            
            nn.Linear(1024, np.prod(img_shape)),
            nn.Tanh(), 
        )

    def forward(self, 
                noise: FloatTensor, 
                labels: IntTensor) -> FloatTensor:
        # noise: float[batch_size x latent_dim]
        # labels: int[batch_size]
        
        # label_emb: [batch_size x label_emb_dim]
        label_emb = self.label_emb(labels)
        
        # inp_G: [batch_size x (latent_dim + label_emb_dim)]
        inp_G = torch.cat([label_emb, noise], dim=-1)

        # img_flat: [batch_size x img_size]
        img_flat = self.seq(inp_G)
        
        # img: [batch_size x C x H x W]
        img = img_flat.view(-1, *self.img_shape)

        return img


class Discriminator(nn.Module):
    def __init__(self,
                 num_classes: int,
                 img_shape: list[int]):
        super().__init__()

        self.img_shape = img_shape

        self.label_emb = nn.Embedding(num_embeddings=num_classes,
                                      embedding_dim=num_classes)

        self.seq = nn.Sequential(
            nn.Linear(num_classes + np.prod(img_shape), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, 
                img_batch: FloatTensor, 
                labels: IntTensor) -> FloatTensor:
        # img_batch: float[batch_size x C x H x W]
        # labels: int[batch_size]
        
        # img_flat: [batch_size x img_size] 
        img_flat = img_batch.view(img_batch.shape[0], -1)
        
        # label_emb: [batch_size x label_emb_dim] 
        label_emb = self.label_emb(labels)
        
        # inp_D: [batch_size x (img_size + label_emb_dim)]
        inp_D = torch.cat([img_flat, label_emb], dim=-1)

        # logits: [batch_size]
        logits = self.seq(inp_D).squeeze(dim=-1)

        return logits 
