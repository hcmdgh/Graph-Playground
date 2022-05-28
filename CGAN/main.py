from util import * 
from dataset.mnist import * 
from .config import * 
from .model import * 
from torchvision.utils import save_image


def generate_images(generator: Generator,
                    epoch: int):
    noise = to_device(torch.normal(mean=0., std=1., size=[NUM_CLASSES ** 2, LATENT_DIM]))
    labels = to_device(torch.tensor([i for _ in range(NUM_CLASSES) for i in range(NUM_CLASSES)], dtype=torch.int64))

    with torch.no_grad():
        fake_img_batch = generator(noise=noise,
                                   labels=labels)

    save_image(fake_img_batch, f"./CGAN/output/epoch_{epoch}.png", nrow=NUM_CLASSES, normalize=True)


def main():
    set_device(DEVICE)
    
    init_log('./log.log')
    
    dataset = load_MNIST_dataset()
    
    img_sample, label_sample = dataset[0]
    
    img_shape = list(img_sample.shape) 
    
    dataloader = DataLoader(
        dataset = dataset, 
        batch_size = BATCH_SIZE,
        shuffle = True,
    )
    
    generator = Generator(num_classes=NUM_CLASSES,
                          latent_dim=LATENT_DIM,
                          img_shape=img_shape)
    discriminator = Discriminator(num_classes=NUM_CLASSES,
                                  img_shape=img_shape)
    generator = to_device(generator)
    discriminator = to_device(discriminator)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))   

    for epoch in range(1, NUM_EPOCHS + 1):
        loss_D_list = []
        loss_G_list = []
        
        for step, (img_batch, labels) in enumerate(tqdm(dataloader), start=1):
            img_batch = to_device(img_batch)
            labels = to_device(labels)
            
            batch_size = img_batch.shape[0]
            
            ones = to_device(torch.ones(batch_size))
            zeros = to_device(torch.zeros(batch_size))
            
            # [BEGIN] Train Generator
            # noise: [batch_size x latent_dim]
            noise = to_device(torch.normal(mean=0., std=1., size=[batch_size, LATENT_DIM]))
            
            # fake_labels: int[batch_size]
            fake_labels = to_device(torch.randint(low=0, high=NUM_CLASSES, size=[batch_size]))
            
            # fake_img_batch: [batch_size x C x H x W]
            fake_img_batch = generator(noise=noise,
                                       labels=fake_labels)
            
            out_D = discriminator(img_batch=fake_img_batch,
                                  labels=fake_labels)
            
            loss_G = F.mse_loss(input=out_D, target=ones)
            
            optimizer_G.zero_grad() 
            loss_G.backward() 
            optimizer_G.step() 

            loss_G_list.append(float(loss_G))
            # [END]
            
            # [BEGIN] Train Discriminator 
            out_D = discriminator(img_batch=img_batch,
                                  labels=labels)
            real_loss_D = F.mse_loss(input=out_D, target=ones)
            
            fake_out_D = discriminator(img_batch=fake_img_batch.detach(),
                                       labels=fake_labels)
            fake_loss_D = F.mse_loss(input=fake_out_D, target=zeros)
            
            loss_D = (real_loss_D + fake_loss_D) / 2
            
            optimizer_D.zero_grad() 
            loss_D.backward() 
            optimizer_D.step() 
            
            loss_D_list.append(float(loss_D))
            # [END]
            
        logging.info(f"epoch: {epoch}, loss_D: {np.mean(loss_D_list):.4f}, loss_G: {np.mean(loss_G_list):.4f}")

        generate_images(generator=generator,
                        epoch=epoch)        


if __name__ == '__main__':
    main() 
