import torch
import numpy as np
import os
import random


def set_seed(seed):
    # Set the random seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

