import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler

class Buffer:

    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.dali = str(config.context_mode) == 'dali_s'
        self.context_window = int(config.context_window) if self.dali else 1
        if self.context_window < 1:
            raise ValueError('buffer.context_window must be positive')
        self.sample_length = self.batch_length + 1
        if self.dali:
            self.sample_length = self.batch_length + self.context_window + 1
        self._buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2), sampler=SliceSampler(num_slices=self.batch_size, end_key=None, traj_key='episode', truncated_key=None, strict_length=True), prefetch=0, batch_size=self.batch_size * self.sample_length)

    def add_transition(self, data):
        self._buffer.extend(data.unsqueeze(1))

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        sample_td = sample_td.view(-1, self.sample_length)
        src_dev = sample_td.device
        if src_dev.type == 'cpu' and self.device.type == 'cuda':
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        if not self.dali:
            initial = (sample_td['stoch'][:, 0], sample_td['deter'][:, 0])
            data = sample_td[:, 1:].clone()
            data.set_('action', sample_td['action'][:, :-1])
            index = [ind.view(-1, self.sample_length)[:, 1:] for ind in info['index']]
            return (data, index, initial, None)
        start = self.context_window
        end = start + self.batch_length
        initial = (sample_td['stoch'][:, start - 1], sample_td['deter'][:, start - 1])
        data = sample_td[:, start:end].clone()
        data.set_('action', sample_td['action'][:, start - 1:end - 1])
        history = sample_td[:, 1:end].clone()
        history.set_('action', sample_td['action'][:, :end - 1])
        context_data = {'history': history, 'target': sample_td[:, start + 1:end + 1].clone(), 'forward_action': sample_td['action'][:, start:end]}
        index = [ind.view(-1, self.sample_length)[:, start:end] for ind in info['index']]
        return (data, index, initial, context_data)

    def update(self, index, stoch, deter):
        index = [ind.reshape(-1) for ind in index]
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        deter = deter.reshape(-1, *deter.shape[2:])
        n = index[0].shape[0]
        self._buffer[index[1], index[0]] = TensorDict({'stoch': stoch, 'deter': deter}, batch_size=(n,))

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
