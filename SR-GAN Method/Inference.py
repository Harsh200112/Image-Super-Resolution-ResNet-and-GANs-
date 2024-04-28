import os
import torch
import numpy as np
from PIL import Image
from Models import SR_ResNet
from torchvision import transforms
import matplotlib.pyplot as plt

channels = 64
small_kernel_size = 3
large_kernel_size = 9
checkpoint_path = 'C:/Users/Harsh Soni/Downloads/CV Project/SR-GAN Method/checkpoint_gan.pt'
low_res_folder = 'C:/Users/Harsh Soni/Downloads/CV Project/Sample Images/low_res'
high_res_folder = 'C:/Users/Harsh Soni/Downloads/CV Project/Sample Images/high_res'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gen = SR_ResNet(large_kernel_size, small_kernel_size, channels, res_blocks=3, scaling_factor=2).to(device)
checkpoint = torch.load(checkpoint_path)
gen.load_state_dict(checkpoint['gen'])

# Get list of low-res and high-res image files
low_res_files = os.listdir(low_res_folder)
high_res_files = os.listdir(high_res_folder)

# Sort the file lists to ensure they are in the same order
low_res_files.sort()
high_res_files.sort()

# Make sure the number of low-res and high-res images are the same
assert len(low_res_files) == len(high_res_files)

# Preprocess the images
transform_low = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

transform_high = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

sample_images = []
for low_res_file, high_res_file in zip(low_res_files, high_res_files):
    low_res_path = os.path.join(low_res_folder, low_res_file)
    high_res_path = os.path.join(high_res_folder, high_res_file)

    low_res_image = Image.open(low_res_path)
    high_res_image = Image.open(high_res_path)

    low_res_image = np.array(low_res_image)[:, :, :3]
    high_res_image = np.array(high_res_image)[:, :, :3]

    low_res_image = transform_low(low_res_image).unsqueeze(0).to(device)
    high_res_image = transform_high(high_res_image).unsqueeze(0).to(device)

    sample_images.append((low_res_image, high_res_image))

gen.eval()

# Perform inference
with torch.no_grad():
    super_res_images = [gen(low_res_image) for low_res_image, _ in sample_images]

# Plot the results
fig, ax = plt.subplots(3, len(sample_images), figsize=(20, 12))
for i, (sample, super_res_image) in enumerate(zip(sample_images, super_res_images)):
    low_res_image, high_res_image = sample
    ax[0, i].imshow(low_res_image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
    ax[1, i].imshow(super_res_image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
    ax[2, i].imshow(high_res_image.squeeze().permute(1, 2, 0).cpu().detach().numpy())
    ax[0, i].set_title("Low")
    ax[1, i].set_title("Super")
    ax[2, i].set_title("High")
    ax[0, i].axis('off')
    ax[1, i].axis('off')
    ax[2, i].axis('off')
plt.show()
