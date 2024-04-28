import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
from Models import SR_ResNet, Discriminator
from DataLoaders import train_loader, val_loader
from Utils import load_checkpoint_gan, save_checkpoint_gan

# Hyperparameters
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
batch_size = 16
epochs = 100
learning_rate = 3e-4
num_classes = 4
small_kernel_size = 3
large_kernel_size = 9
channels = 64
in_channels = 3
out_channels = 3
scaling_factor = 2
res_blocks = 16
img_size = 256
num_workers = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss Functions
def get_gen_loss(disc_fakes):
    return torch.sum(torch.log(1 - disc_fakes + 1e-6))

def get_disc_loss(disc_reals, disc_fakes):
    return -1 * torch.sum(torch.log(disc_reals + 1e-6) + torch.log(1 - disc_fakes + 1e-6))

class vggL(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:25].eval().to(device)
        self.loss = nn.MSELoss()

    def forward(self, first, second):
        vgg_first = self.vgg(first)
        vgg_second = self.vgg(second)
        perceptual_loss = self.loss(vgg_first, vgg_second)
        return perceptual_loss

# Training the GAN Model
vgg_loss = vggL()
def train_gen(gen, disc, gen_opt, lr, hr):
    gen.train()
    gen_opt.zero_grad()
    sr = gen(lr)
    disc_fake = disc(sr)
    gen_loss = get_gen_loss(disc_fake)
    content_loss = 0.006 * vgg_loss(sr, hr)
    adversarial_loss = 1e-3 * gen_loss
    total_loss = adversarial_loss + content_loss
    total_loss.backward()
    gen_opt.step()
    return total_loss.item()

def train_disc(disc, gen, disc_opt, lr, hr):
    disc.train()
    disc_opt.zero_grad()
    real_pred = disc(hr)
    fake_data = gen(lr)
    fake_pred = disc(fake_data.detach())
    disc_loss = get_disc_loss(real_pred, fake_pred)
    disc_loss.backward()
    disc_opt.step()
    return disc_loss.item()

checkpoint_dir_gan = 'checkpoints_gan'
os.makedirs(checkpoint_dir_gan, exist_ok=True)

gen = SR_ResNet(large_kernel_size, small_kernel_size, channels, res_blocks=3, scaling_factor=2).to(device)
disc = Discriminator(3, 64).to(device)
gen_opt = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
disc_opt = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

start_epoch, gen_losses, disc_losses = load_checkpoint_gan(gen, disc, gen_opt, disc_opt, checkpoint_dir_gan)
for epoch in range(epochs):
    curr_gen_loss = 0
    curr_disc_loss = 0
    for batch_idx, (lr, hr) in enumerate(train_loader):
        hr = hr.to(device)
        lr = lr.to(device)
        disc_loss = train_disc(disc, gen, disc_opt, lr, hr)
        gen_loss = train_gen(gen, disc, gen_opt, lr, hr)
        curr_gen_loss += gen_loss
        curr_disc_loss += disc_loss
        
    gen_losses.append(curr_gen_loss/len(train_loader))
    disc_losses.append(curr_disc_loss/len(train_loader))
    
    print(f"Epoch [{epoch+1}/{epochs}] Loss D: {disc_losses[-1]:.4f}, Loss G: {gen_losses[-1]:.4f}")
    save_checkpoint_gan(epoch, gen, disc, gen_opt, disc_opt, gen_losses, disc_losses, checkpoint_dir_gan)