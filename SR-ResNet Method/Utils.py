import os
import torch

def save_checkpoint(epoch, model, optimizer, t_loss, v_loss, root_dir):
    checkpoint = os.path.join(root_dir, 'checkpoint.pt')
    torch.save({
        'epoch' : epoch,
        't_loss' : t_loss,
        'v_loss' : v_loss,
        'model' : model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, checkpoint)
    print(f'Checkpoint Saved at epoch{epoch+1}')

def load_checkpoint(model, optimizer, root_dir):
    checkpoint = os.path.join(root_dir, 'checkpoint.pt')
    if os.path.exists(checkpoint):
        checkpoint_file = torch.load(checkpoint)
        epochs = checkpoint_file['epoch']
        t_loss = checkpoint_file['t_loss']
        v_loss = checkpoint_file['v_loss']
        model = model.load_state_dict(checkpoint_file['model'])
        optimizer = optimizer.load_state_dict(checkpoint_file['optimizer'])
        return epochs + 1, t_loss, v_loss
    
    else:
        return 0, [], []