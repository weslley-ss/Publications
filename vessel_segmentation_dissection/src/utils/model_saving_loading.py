import argparse
import os
import os.path as osp
import torch

def save_model(path, model, optimizer, seed, stats= None,):
    os.makedirs(path, exist_ok=True)
    model_name = f"model_checkpoint{seed}.pth"
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats':stats
            }, osp.join(path, model_name))
    print(f"####################### save model {model_name}")

def load_model(model, experiment_path, seed, device='cuda', with_opt=False):
    model_name = f"model_checkpoint{seed}.pth"
    checkpoint_path = osp.join(experiment_path, model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if with_opt:
        return model, checkpoint['stats'], checkpoint['optimizer_state_dict']
    print(f"####################### load model {model_name}")
    return model, checkpoint['stats'], None

def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
        return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')
