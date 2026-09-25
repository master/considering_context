import torch
from torch.optim import Optimizer

class LaProp(Optimizer):

    def __init__(self, params, lr=0.0004, betas=(0.9, 0.999), eps=1e-15):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_lr_1'] = 0.0
                    state['exp_avg_lr_2'] = 0.0
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = (state['exp_avg'], state['exp_avg_sq'])
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                state['exp_avg_lr_1'] = state['exp_avg_lr_1'] * beta1 + (1 - beta1) * group['lr']
                state['exp_avg_lr_2'] = state['exp_avg_lr_2'] * beta2 + (1 - beta2)
                bias_correction1 = state['exp_avg_lr_1'] / group['lr'] if group['lr'] != 0.0 else 1.0
                step_size = 1 / bias_correction1
                bias_correction2 = state['exp_avg_lr_2']
                denom = exp_avg_sq
                denom = denom.div(bias_correction2).sqrt_().add_(group['eps'])
                step_of_this_grad = grad / denom
                exp_avg.mul_(beta1).add_((1 - beta1) * group['lr'], step_of_this_grad)
                p.data.add_(-step_size, exp_avg)
