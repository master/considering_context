import torch
from torch import distributions as torchd
from torch import nn
import distributions as dists
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_

class BlockGRUDeter(nn.Module):

    def __init__(self, deter, stoch, act_dim, hidden, blocks, dynlayers, context_dim=0, act='SiLU'):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(nn.Linear(deter, hidden, bias=True), nn.RMSNorm(hidden, eps=0.0001, dtype=torch.float32), act())
        self._dyn_in1 = nn.Sequential(nn.Linear(stoch, hidden, bias=True), nn.RMSNorm(hidden, eps=0.0001, dtype=torch.float32), act())
        self._dyn_in2 = nn.Sequential(nn.Linear(act_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=0.0001, dtype=torch.float32), act())
        self._dyn_in3 = None
        if context_dim:
            self._dyn_in3 = nn.Sequential(nn.Linear(context_dim, hidden, bias=True), nn.RMSNorm(hidden, eps=0.0001, dtype=torch.float32), act())
        self._dyn_hid = nn.Sequential()
        projected_inputs = 3 + int(self._dyn_in3 is not None)
        in_ch = (projected_inputs * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f'dyn_hid_{i}', BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(f'norm_{i}', nn.RMSNorm(deter, eps=0.0001, dtype=torch.float32))
            self._dyn_hid.add_module(f'act_{i}', act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.state_size = deter
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action, context=None):
        B = action.shape[0]
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)
        inputs = [x0, x1, x2]
        if self._dyn_in3 is not None:
            if context is None:
                raise ValueError('the deterministic transition requires context')
            inputs.append(self._dyn_in3(context))
        x = torch.cat(inputs, -1)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))
        x = self._dyn_hid(x)
        x = self._dyn_gru(x)
        gates = torch.chunk(self.flat2group(x), 3, dim=-1)
        reset, cand, update = (self.group2flat(x) for x in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        return update * cand + (1 - update) * deter

    def initial(self, batch_size, device):
        return torch.zeros(batch_size, self.state_size, dtype=torch.float32, device=device)

    def output(self, state):
        return state

class RSSM(nn.Module):

    def __init__(self, config, embed_size, act_dim, transition_context_dim=0, posterior_context_dim=0, carry_state=False):
        super().__init__()
        self.carry_state = bool(carry_state)
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self.transition_context_dim = int(transition_context_dim)
        self.posterior_context_dim = int(posterior_context_dim)
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        self._deter_net = BlockGRUDeter(self._deter, self.flat_stoch, act_dim, self._hidden, blocks=self._blocks, dynlayers=self._dyn_layers, context_dim=self.transition_context_dim, act=config.act)
        self.deter_state_size = self._deter_net.state_size
        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size + self.posterior_context_dim
        for i in range(self._obs_layers):
            self._obs_net.add_module(f'obs_net_{i}', nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f'obs_net_n_{i}', nn.RMSNorm(self._hidden, eps=0.0001, dtype=torch.float32))
            self._obs_net.add_module(f'obs_net_a_{i}', act())
            inp_dim = self._hidden
        self._obs_net.add_module('obs_net_logit', nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module('obs_net_lambda', LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)))
        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f'img_net_{i}', nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f'img_net_n_{i}', nn.RMSNorm(self._hidden, eps=0.0001, dtype=torch.float32))
            self._img_net.add_module(f'img_net_a_{i}', act())
            inp_dim = self._hidden
        self._img_net.add_module('img_net_logit', nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module('img_net_lambda', LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)))
        self.apply(weight_init_)

    def initial(self, batch_size):
        deter = self._deter_net.initial(batch_size, self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        return (stoch, deter)

    def observe(self, embed, action, initial, reset, context=None):
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = ([], [], [])
        for i in range(L):
            step_context = context[:, i] if context is not None else None
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i], embed[:, i], reset[:, i], step_context)
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        return (stochs, deters, logits)

    def obs_step(self, stoch, deter, prev_action, embed, reset, context=None):
        if not self.carry_state:
            stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
            deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action)
        transition_context = context if self.transition_context_dim else None
        posterior_context = context if self.posterior_context_dim else None
        if (self.transition_context_dim or self.posterior_context_dim) and context is None:
            raise ValueError('the RSSM conditioning path requires context')
        deter = self._deter_net(stoch, deter, prev_action, transition_context)
        inputs = [self.deter_output(deter), embed]
        if posterior_context is not None:
            inputs.append(posterior_context)
        x = torch.cat(inputs, dim=-1)
        logit = self._obs_net(x)
        stoch = self.get_dist(logit).rsample()
        return (stoch, deter, logit)

    def img_step(self, stoch, deter, prev_action, context=None):
        if self.transition_context_dim and context is None:
            raise ValueError('the RSSM transition requires context')
        transition_context = context if self.transition_context_dim else None
        deter = self._deter_net(stoch, deter, prev_action, transition_context)
        stoch, _ = self.prior(deter)
        return (stoch, deter)

    def prior(self, deter):
        logit = self._img_net(self.deter_output(deter))
        stoch = self.get_dist(logit).rsample()
        return (stoch, logit)

    def get_feat(self, stoch, deter):
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        return torch.cat([stoch, self.deter_output(deter)], -1)

    def deter_output(self, deter):
        return self._deter_net.output(deter)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return (dyn_loss, rep_loss)
