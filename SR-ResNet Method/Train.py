import os
import torch
import torch.nn as nn
import torch.optim as optim
from Models import SR_ResNet
from Utils import load_checkpoint, save_checkpoint
from DataLoaders import train_loader, val_loader

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training SR-ResNet
model = SR_ResNet(large_kernel_size, small_kernel_size, channels, res_blocks=3, scaling_factor=2).to(device)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

checkpoint_dir_resnet = 'checkpoints_resnet'
os.makedirs(checkpoint_dir_resnet, exist_ok=True)

start_epoch, t_loss, v_loss = load_checkpoint(model, optimizer, checkpoint_dir_resnet)
best_val_loss = float("inf")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (lr, hr) in enumerate(train_loader):
        hr = hr.to(device)
        lr = lr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss /= len(train_loader)
    t_loss.append(train_loss)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (lr, hr) in enumerate(val_loader):
            hr = hr.to(device)
            lr = lr.to(device)
            sr = model(lr)
            valid_loss = criterion(sr, hr)
            val_loss += valid_loss.item()
        val_loss = val_loss/len(val_loader)
        v_loss.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"checkpoints_resnet/model.pth")
                
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}')
    print(f'Epoch {epoch+1}, Val Loss: {val_loss}')
    save_checkpoint(epoch, model, optimizer, t_loss, v_loss, checkpoint_dir_resnet)