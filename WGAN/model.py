from util import * 


class Generator(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 img_shape: list[int]):
        super().__init__()
        
        self.img_shape = img_shape 

        self.seq = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, np.prod(img_shape)),
            nn.Tanh(),
        )
        
    def forward(self, inp: FloatTensor) -> FloatTensor:
        # inp: float[batch_size x latent_dim]
        
        # img: [batch_size x img_shape]
        img = self.seq(inp)
        
        # img: [batch_size x C x H x W]
        img = img.view(-1, *self.img_shape)

        return img


class Discriminator(nn.Module):
    def __init__(self,
                 img_shape: list[int]):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, img: FloatTensor) -> FloatTensor:
        # img: float[batch_size x C x H x W]
        
        # img_flat: [batch_size x img_size]
        img_flat = img.view(img.shape[0], -1)

        # logit: [batch_size]
        logit = self.seq(img_flat).squeeze(dim=-1)

        return logit 
