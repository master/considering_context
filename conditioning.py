import torch
from torch import nn
from tools import weight_init_

class DALIContextEncoder(nn.Module):

    def __init__(self, observation_dim, action_dim, width, heads, context_dim, window):
        super().__init__()
        self.window = int(window)
        self.heads = int(heads)
        self.input = nn.Linear(int(observation_dim) + int(action_dim), int(width))
        self.norm1 = nn.LayerNorm(int(width))
        self.attention = nn.MultiheadAttention(int(width), self.heads, batch_first=True)
        self.norm2 = nn.LayerNorm(int(width))
        self.feedforward = nn.Sequential(nn.Linear(int(width), int(width)), nn.SiLU(), nn.Linear(int(width), int(width)))
        self.output = nn.Linear(int(width), int(context_dim))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight_init_(module)

    def forward(self, observation, previous_action, is_first):
        reset = self._validate_inputs(observation, previous_action, is_first)
        previous_action = torch.where(reset.unsqueeze(-1), torch.zeros_like(previous_action), previous_action)
        return self._encode(observation, previous_action, self._attention_mask(reset))

    def encode_window(self, observation, previous_action):
        if observation.shape[:-1] != previous_action.shape[:-1]:
            raise ValueError('DALI observations and previous actions must have matching batch and time dimensions')
        if observation.ndim != 3:
            raise ValueError('DALI context inference expects tensors shaped (batch, time, features)')
        if observation.shape[1] != self.window:
            raise ValueError(f'DALI complete windows must contain exactly {self.window} steps')
        return self._encode(observation, previous_action, None)[:, -1]

    def _validate_inputs(self, observation, previous_action, is_first):
        if observation.shape[:-1] != previous_action.shape[:-1]:
            raise ValueError('DALI observations and previous actions must have matching batch and time dimensions')
        if observation.ndim != 3:
            raise ValueError('DALI context inference expects tensors shaped (batch, time, features)')
        reset = is_first.bool()
        if reset.ndim == 3 and reset.shape[-1] == 1:
            reset = reset.squeeze(-1)
        if reset.shape != observation.shape[:2]:
            raise ValueError('DALI reset flags must match the observation batch and time dimensions')
        return reset

    def _encode(self, observation, previous_action, mask):
        x = self.input(torch.cat([observation, previous_action], dim=-1))
        skip = x
        x = self.norm1(x)
        x = self.attention(x, x, x, attn_mask=mask, need_weights=False)[0] + skip
        skip = x
        x = self.feedforward(self.norm2(x)) + skip
        return self.output(x)

    def _attention_mask(self, reset):
        batch, length = reset.shape
        position = torch.arange(length, device=reset.device)
        query = position[:, None]
        key = position[None, :]
        allowed = (key <= query) & (key > query - self.window)
        episode = torch.cumsum(reset.to(torch.int64), dim=1)
        allowed = allowed.unsqueeze(0) & (episode[:, :, None] == episode[:, None, :])
        mask = ~allowed
        return mask[:, None].expand(batch, self.heads, length, length).reshape(batch * self.heads, length, length)

class ForwardDynamics(nn.Module):

    def __init__(self, observation_dim, action_dim, context_dim, units, layers):
        super().__init__()
        input_dim = int(observation_dim) + int(action_dim) + int(context_dim)
        modules = []
        for _ in range(int(layers)):
            modules.extend([nn.Linear(input_dim, int(units)), nn.SiLU()])
            input_dim = int(units)
        modules.append(nn.Linear(input_dim, int(observation_dim)))
        self.network = nn.Sequential(*modules)
        self.network.apply(weight_init_)

    def forward(self, observation, action, context):
        return self.network(torch.cat([observation, action, context], dim=-1))

class EpisodeSummaryEncoder(nn.Module):

    def __init__(self, observation_dim, action_dim, units, layers, context_dim):
        super().__init__()
        self.units = int(units)
        input_dim = 2 * int(observation_dim) + int(action_dim)
        modules = []
        for _ in range(int(layers)):
            modules.extend([nn.Linear(input_dim, int(units)), nn.SiLU()])
            input_dim = int(units)
        self.features = nn.Sequential(*modules)
        self.summary = nn.Sequential(nn.Linear(input_dim, int(units)), nn.SiLU(), nn.Linear(int(units), int(context_dim)), nn.Tanh())
        self.apply(weight_init_)

    def transition_features(self, previous_observation, previous_action, observation):
        return self.features(torch.cat([previous_observation, previous_action, observation], dim=-1))

    def summarize(self, pooled):
        return self.summary(pooled)

    def encode_segment(self, previous_observation, previous_action, observation, is_first):
        reset = is_first.bool()
        if reset.ndim == 3 and reset.shape[-1] == 1:
            reset = reset.squeeze(-1)
        if observation.ndim != 3 or reset.shape != observation.shape[:2]:
            raise ValueError('episode summaries expect (batch, time, features) tensors with matching reset flags')
        features = self.transition_features(previous_observation, previous_action, observation)
        length = reset.shape[1]
        valid = ~reset & (torch.arange(length, device=reset.device) > 0)[None, :]
        segment = torch.cumsum(reset.to(torch.int64), dim=1)
        same = (segment[:, :, None] == segment[:, None, :]) & valid[:, None, :]
        weights = same.to(features.dtype)
        pooled = weights @ features / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return self.summarize(pooled)

class EpisodeChain(nn.Module):

    def __init__(self, context_dim, units):
        super().__init__()
        self.context_dim = int(context_dim)
        self.units = int(units)
        self.gru = nn.GRU(self.context_dim, self.units, batch_first=True)
        self.output = nn.Linear(self.units, self.context_dim)
        weight_init_(self.output)

    def step(self, latent, hidden):
        output, hidden = self.gru(latent[:, None], hidden[None].contiguous())
        return (torch.tanh(self.output(output[:, 0])), hidden[0])

    def initial(self, batch):
        device = self.output.weight.device
        return self.step(torch.zeros(batch, self.context_dim, device=device), torch.zeros(batch, self.units, device=device))

    def forward(self, latents):
        inputs = torch.cat([torch.zeros_like(latents[:, :1]), latents[:, :-1]], dim=1)
        output, _ = self.gru(inputs)
        return torch.tanh(self.output(output))

class EpisodeLatentCache(nn.Module):

    def __init__(self, env_num, max_episodes, context_dim):
        super().__init__()
        self.env_num = int(env_num)
        self.max_episodes = int(max_episodes)
        if self.env_num < 1 or self.max_episodes < 1:
            raise ValueError('the episode latent cache needs at least one environment and one episode slot')
        shape = (self.env_num, self.max_episodes)
        self.register_buffer('latent', torch.zeros(*shape, int(context_dim)), persistent=False)
        self.register_buffer('prediction', torch.zeros(*shape, int(context_dim)), persistent=False)
        self.register_buffer('valid', torch.zeros(shape, dtype=torch.bool), persistent=False)
        self.register_buffer('predicted', torch.zeros(shape, dtype=torch.bool), persistent=False)
        self.register_buffer('prediction_episode', torch.zeros(shape, dtype=torch.int64), persistent=False)
        self.register_buffer('count', torch.zeros(self.env_num, dtype=torch.int64), persistent=False)

    def _slot(self, episode):
        return episode.long() % self.max_episodes

    def is_current(self, env, episode):
        return episode.long() + self.max_episodes >= self.count[env.long()]

    def _write(self, values, flags, env, episode, value, mask):
        env = env.long()
        slot = self._slot(episode)
        mask = mask.bool()
        values[env, slot] = torch.where(mask[:, None], value.to(values.dtype), values[env, slot])
        flags[env, slot] = flags[env, slot] | mask

    def write_latent(self, env, episode, latent, mask):
        self._write(self.latent, self.valid, env, episode, latent, mask)
        self.count.scatter_reduce_(0, env.long(), torch.where(mask.bool(), episode.long() + 1, 0), reduce='amax')

    def write_prediction(self, env, episode, prediction, mask):
        self._write(self.prediction, self.predicted, env, episode, prediction, mask)
        env, slot = (env.long(), self._slot(episode))
        self.prediction_episode[env, slot] = torch.where(mask.bool(), episode.long(), self.prediction_episode[env, slot])

    def read_prediction(self, env, episode):
        env = env.long()
        slot = self._slot(episode)
        fresh = self.predicted[env, slot] & (self.prediction_episode[env, slot] == episode.long())
        return (self.prediction[env, slot], fresh)

    def recent(self, length):
        length = int(length)
        if length > self.max_episodes:
            raise ValueError('the chain window cannot exceed the number of cached episodes')
        start = (self.count - length).clamp_min(0)
        episode = start[:, None] + torch.arange(length, device=self.count.device)[None, :]
        slot = episode % self.max_episodes
        env = torch.arange(self.env_num, device=self.count.device)[:, None]
        valid = self.valid[env, slot] & (episode < self.count[:, None])
        return (self.latent[env, slot], valid)
