import torch
import numpy as np

alpha = 1e-6

def min_max_norm(x):
    # judge if x is a tensor or a numpy array
    if isinstance(x, torch.Tensor):
        x = x - torch.min(x)
        x = x / (torch.max(x) - torch.min(x) + alpha)
    else:
        x = x - np.min(x)
        x = x / (np.max(x) - np.min(x) + alpha)
    return x

def mean_var_norm(x):
    # judge if x is a tensor or a numpy array
    if isinstance(x, torch.Tensor):
        x = x - torch.mean(x)
        x = x / (torch.std(x) + alpha)
    else:
        x = x - np.mean(x)
        x = x / (np.std(x) + alpha)
    return x