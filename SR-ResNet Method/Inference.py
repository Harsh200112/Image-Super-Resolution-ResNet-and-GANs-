import os
import torch
from Models import SR_ResNet
import matplotlib.pyplot as plt
from DataLoaders import val_loader

channels = 64
small_kernel_size = 3
large_kernel_size = 9
checkpoint_path = 'C:/Users/Harsh Soni/Downloads/CV Project/SR-ResNet Method/checkpoint_SR-ResNet.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SR_ResNet(large_kernel_size, small_kernel_size, channels, res_blocks=3, scaling_factor=2).to(device)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])

low_res = next(iter(val_loader))[0].to(device)
high_res = next(iter(val_loader))[1].to(device)

model.eval()
super_res = model(low_res)

fig, ax = plt.subplots(3, 5, figsize=(20, 12))
ax[0, 0].set_title("Low")
ax[1, 0].set_title("Super")
ax[2, 0].set_title("High")
for i in range(5):
    ax[0, i].imshow(low_res[i].permute(1, 2, 0).cpu().detach().numpy())
    ax[1, i].imshow(super_res[i].permute(1, 2, 0).cpu().detach().numpy())
    ax[2, i].imshow(high_res[i].permute(1, 2, 0).cpu().detach().numpy())
    ax[0, i].axis('off')
    ax[1, i].axis('off')
    ax[2, i].axis('off')
plt.show()