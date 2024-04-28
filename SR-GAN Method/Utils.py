import os
import torch

def save_checkpoint_gan(epoch, gen, disc, gen_opt, disc_opt, gen_losses, disc_losses, root_dir):
    checkpoint_gan = os.path.join(root_dir, 'checkpoint_gan.pt')
    torch.save({
        'epoch' : epoch,
        'gen_losses' : gen_losses,
        'disc_losses' : disc_losses,
        'gen' : gen.state_dict(),
        'disc' : disc.state_dict(),
        'gen_opt': gen_opt.state_dict(),
        'disc_opt': disc_opt.state_dict(),
    }, checkpoint_gan)
    print(f'Checkpoint Saved at epoch{epoch+1}')

def load_checkpoint_gan(gen, disc, gen_opt, disc_opt, root_dir):
    checkpoint_gan = os.path.join(root_dir, 'checkpoint_gan.pt')
    if os.path.exists(checkpoint_gan):
        checkpoint_file_gan = torch.load(checkpoint_gan)
        epochs = checkpoint_file_gan['epoch']
        gen_losses = checkpoint_file_gan['gen_losses']
        disc_losses = checkpoint_file_gan['disc_losses']
        gen = gen.load_state_dict(checkpoint_file_gan['gen'])
        disc = disc.load_state_dict(checkpoint_file_gan['disc'])
        gen_opt = gen_opt.load_state_dict(checkpoint_file_gan['gen_opt'])
        disc_opt = disc_opt.load_state_dict(checkpoint_file_gan['disc_opt'])
        return epochs + 1, gen_losses, disc_losses
    
    else:
        return 0, [], []