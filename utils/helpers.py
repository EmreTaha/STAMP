import random
import numpy as np
import torch
import torch.nn as nn
import io
import requests
import monai


def initialize(seed=3407, allow_tf32=False, deterministic=True):
    torch.cuda.empty_cache()

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    #torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    np.random.seed(seed)

def worker_init_fn(worker_id):                                                          
    np.random.seed(torch.initial_seed() // 2**32 + worker_id)
    random.seed(torch.initial_seed() // 2**32 + worker_id)


def get_BiT_weights(bit_variant='BiT-M-R50x1-CIFAR10'):
    #TODO this doesnt belong here!
    response = requests.get(f'https://storage.googleapis.com/bit_models/{bit_variant}.npz')
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))
