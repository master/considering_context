import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import init as nn_init

def to_np(x):
    return x.detach().cpu().numpy()

def to_f32(x):
    return x.to(dtype=torch.float32)

def to_i32(x):
    return x.to(dtype=torch.int32)

def weight_init_(m, fan_type='in'):
    if isinstance(m, nn.RMSNorm):
        with torch.no_grad():
            m.weight.fill_(1.0)
        return
    weight = getattr(m, 'weight', None)
    if weight is None:
        return
    if weight.numel() == 0:
        return
    in_num, out_num = nn_init._calculate_fan_in_and_fan_out(weight)
    with torch.no_grad():
        fan = {'avg': (in_num + out_num) / 2, 'in': in_num, 'out': out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan)
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
        bias = getattr(m, 'bias', None)
        if bias is not None:
            bias.fill_(0.0)

def convert(value, precision=32):
    if isinstance(value, dict):
        return {key: convert(val) for key, val in value.items()}
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
        dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision]
    elif np.issubdtype(value.dtype, np.signedinteger):
        dtype = {16: np.int16, 32: np.int32, 64: np.int64}[precision]
    elif np.issubdtype(value.dtype, np.uint8):
        dtype = np.uint8
    elif np.issubdtype(value.dtype, bool):
        dtype = bool
    else:
        raise NotImplementedError(value.dtype)
    return value.astype(dtype)

class Every:

    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if not self._every:
            return 0
        if self._last is None:
            self._last = step
            return 1
        count = int((step - self._last) / self._every)
        self._last += self._every * count
        return count

class Once:

    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def enable_deterministic_run():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def rpad(x, pad):
    for _ in range(pad):
        x = x.unsqueeze(-1)
    return x
