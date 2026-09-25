import copy
import re
from functools import partial
import torch
from torch import nn
import distributions as dists
from tools import weight_init_

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BlockLinear(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float=1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)
        self.weight = nn.Parameter(torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks))
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-1]
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)
        x = torch.einsum('...gi,oig->...go', x, self.weight)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias

class MultiEncoder(nn.Module):

    def __init__(self, config, shapes):
        super().__init__()
        excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
        shapes = {k: v for k, v in shapes.items() if k not in excluded and (not k.startswith('log_'))}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}
        self.out_dim = 0
        self.selectors = []
        self.encoders = []
        if self.mlp_shapes:
            inp_dim = sum([sum(v) for v in self.mlp_shapes.values()])
            self.encoders.append(MLP(config.mlp, inp_dim))
            self.selectors.append(lambda obs: torch.cat([obs[k] for k in self.mlp_shapes], -1))
            self.out_dim += self.encoders[-1].out_dim
        self.encoders = nn.ModuleList(self.encoders)
        if len(self.encoders) == 1:
            self.fuser = lambda x: x[0]
        else:
            raise NotImplementedError
        self.apply(weight_init_)

    def forward(self, obs):
        return self.fuser([enc(sel(obs)) for enc, sel in zip(self.encoders, self.selectors)])

class MultiDecoder(nn.Module):

    def __init__(self, config, deter, flat_stoch, shapes, condition_dim=0):
        super().__init__()
        self.condition_dim = int(condition_dim)
        excluded = ('is_first', 'is_last', 'is_terminal')
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.mlp_shapes = {k: v for k, v in shapes.items() if len(v) in (1, 2) and re.match(config.mlp_keys, k)}
        self.all_keys = list(self.mlp_shapes.keys())
        if self.mlp_shapes:
            shape = (sum((sum(x) for x in self.mlp_shapes.values())),)
            mlp_config = copy.deepcopy(config.mlp)
            mlp_config.shape = shape
            self._mlp = MLPHead(mlp_config, deter + flat_stoch + self.condition_dim)
            self._mlp_dist = partial(getattr(dists, str(config.mlp_dist.name)), **config.mlp_dist)

    def forward(self, stoch, deter, condition=None):
        flat_stoch = stoch.reshape(*deter.shape[:-1], -1)
        if self.condition_dim:
            if condition is None:
                raise ValueError('the decoder requires context')
            flat_stoch = torch.cat([flat_stoch, condition], dim=-1)
        dists = {}
        if self.mlp_shapes:
            split_sizes = [v[0] for v in self.mlp_shapes.values()]
            feat = torch.cat([flat_stoch, deter], -1)
            outputs = self._mlp(feat)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update({key: self._mlp_dist(output) for key, output in zip(self.mlp_shapes.keys(), outputs)})
        return dists

class MLP(nn.Module):

    def __init__(self, config, inp_dim):
        super().__init__()
        act = getattr(torch.nn, config.act)
        self._symlog_inputs = bool(config.symlog_inputs)
        self._device = torch.device(config.device)
        self.layers = nn.Sequential()
        for i in range(config.layers):
            self.layers.add_module(f'{config.name}_linear{i}', nn.Linear(inp_dim, config.units, bias=True))
            self.layers.add_module(f'{config.name}_norm{i}', nn.RMSNorm(config.units, eps=0.0001, dtype=torch.float32))
            self.layers.add_module(f'{config.name}_act{i}', act())
            inp_dim = config.units
        self.out_dim = config.units

    def forward(self, x):
        if self._symlog_inputs:
            x = dists.symlog(x)
        return self.layers(x)

class MLPHead(nn.Module):

    def __init__(self, config, inp_dim):
        super().__init__()
        self.mlp = MLP(config, inp_dim)
        self._dist_name = str(config.dist.name)
        self._outscale = float(config.outscale)
        self._dist = getattr(dists, str(config.dist.name))
        if self._dist_name == 'bounded_normal':
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0] * 2, bias=True)
            kwargs = {'min_std': float(config.dist.min_std), 'max_std': float(config.dist.max_std)}
        elif self._dist_name == 'symexp_twohot':
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {'device': torch.device(config.device), 'bin_num': int(config.dist.bin_num)}
        elif self._dist_name in ('binary', 'identity'):
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {}
        else:
            raise NotImplementedError
        self._dist = partial(self._dist, **kwargs)
        self.mlp.apply(weight_init_)
        self.last.apply(weight_init_)
        if self._outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(self._outscale)

    def forward(self, x):
        return self._dist(self.last(self.mlp(x)))

class ReturnEMA(nn.Module):

    def __init__(self, device, alpha=0.01):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.register_buffer('ema_vals', torch.zeros(2, dtype=torch.float32, device=self.device))

    def __call__(self, x):
        x_quantile = torch.quantile(torch.flatten(x.detach()), self.range)
        self.ema_vals.copy_(self.alpha * x_quantile.detach() + (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return (offset.detach(), scale.detach())
