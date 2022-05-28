from util import * 
from dataset.mnist import * 
from .model import * 
from .config import * 
from torchvision.utils import save_image


def main():
    set_device(DEVICE)
    
    init_log('./log.log')
    
    mnist = load_MNIST_dataset()

    img_sample, label_sample = mnist[0]
    
    img_shape = list(img_sample.shape)
    
    generator = Generator(latent_dim=LATENT_DIM,
                          img_shape=img_shape)
    discriminator = Discriminator(img_shape=img_shape)
    generator = to_device(generator)
    discriminator = to_device(discriminator)

    dataloader = DataLoader(
        dataset = mnist,
        batch_size = BATCH_SIZE,
        shuffle = True,
    )
    
    optimizer_G = optim.RMSprop(generator.parameters(), lr=LR)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=LR)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        for step, (img_batch, _) in enumerate(dataloader, start=1):
            # img_batch: [batch_size x C x H x W]
            img_batch = to_device(img_batch)
            
            batch_size = img_batch.shape[0]
            
            # 生成噪声，作为生成器的输入
            # noise: [batch_size x latent_dim]
            noise = to_device(torch.normal(mean=0., std=1., size=[batch_size, LATENT_DIM]))
    
            # fake_img_batch: [batch_size x C x H x W]
            fake_img_batch = generator(noise).detach() 
            
            loss_D = - torch.mean(discriminator(img_batch)) \
                     + torch.mean(discriminator(fake_img_batch))
                     
            optimizer_D.zero_grad() 
            loss_D.backward() 
            optimizer_D.step() 
            
            # 判别器权重范围约束
            for parameter in discriminator.parameters():
                parameter.detach().clamp_(-CLIP_VAL, CLIP_VAL)
                
            if step % NUM_STEPS_FOR_GENERATOR_BACKWARD == 0:
                fake_img_batch = generator(noise)
                
                loss_G = -torch.mean(discriminator(fake_img_batch))
                
                optimizer_G.zero_grad()
                loss_G.backward() 
                optimizer_G.step() 
                
                logging.info(f"epoch: {epoch}, step: {step}, loss_D: {float(loss_D):.4f}, loss_G: {float(loss_G):.4f}")                

        save_image(
            fake_img_batch[:25],
            fp = f'./WGAN/output/epoch_{epoch}.png',
            nrow = 5,
            normalize = True, 
        )


if __name__ == '__main__':
    main() 
