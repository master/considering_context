import copy
from collections import OrderedDict
import torch
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
import conditioning
import networks
import rssm
from optim import LaProp, clip_grad_agc_
from tools import to_f32

class Dreamer(nn.Module):

    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = sum(act_space.shape)
        self.context_mode = str(config.context.mode)
        self.context_key = str(config.context.key)
        valid_context_modes = {'hidden', 'carried', 'crssm', 'dali_s', 'lilac'}
        if self.context_mode not in valid_context_modes:
            raise ValueError(f'unknown context mode: {self.context_mode}')
        self._uses_true_context = self.context_mode == 'crssm'
        self._uses_dali = self.context_mode == 'dali_s'
        self._uses_lilac = self.context_mode == 'lilac'
        self._deep_context = self.context_mode == 'crssm'
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        physical_shapes = {key: shape for key, shape in shapes.items() if key != self.context_key}
        self.encoder = networks.MultiEncoder(config.encoder, physical_shapes)
        self.embed_size = self.encoder.out_dim
        self._normalize_true_context = bool(config.context.normalize)
        context_dim = 0
        if self._uses_true_context:
            if self.context_key not in shapes or len(shapes[self.context_key]) != 1:
                raise ValueError(f"{self.context_mode} requires a one-dimensional '{self.context_key}' observation")
            context_dim = shapes[self.context_key][0]
            if self._normalize_true_context:
                if config.context.low is None or config.context.high is None:
                    raise ValueError('normalized context requires context.low and context.high')
                low = torch.as_tensor(config.context.low, dtype=torch.float32, device=self.device)
                high = torch.as_tensor(config.context.high, dtype=torch.float32, device=self.device)
                if low.shape != (context_dim,) or high.shape != (context_dim,) or (not torch.all(high > low)):
                    raise ValueError('context.low and context.high must define valid bounds for every context dimension')
                self.register_buffer('_context_low_buffer', low)
                self.register_buffer('_context_high_buffer', high)
        self.context_encoder = None
        self.forward_dynamics = None
        if self._uses_dali:
            dali = config.context.dali
            self._dali_observation_key = str(dali.observation_key)
            if self._dali_observation_key not in physical_shapes:
                raise ValueError(f"DALI input '{self._dali_observation_key}' is not present in the observation space")
            dali_shape = physical_shapes[self._dali_observation_key]
            if len(dali_shape) != 1:
                raise ValueError('state-based DALI requires a one-dimensional observation')
            dali_observation_dim = dali_shape[0]
            self._dali_history_shapes = {self._dali_observation_key: dali_shape}
            context_dim = int(dali.context_dim)
            self._dali_window = int(config.context.window)
            if self._dali_window < 1:
                raise ValueError('DALI context.window must be positive')
            self.context_encoder = conditioning.DALIContextEncoder(dali_observation_dim, self.act_dim, dali.width, dali.heads, context_dim, self._dali_window)
            self.forward_dynamics = conditioning.ForwardDynamics(dali_observation_dim, self.act_dim, context_dim, dali.forward_units, dali.forward_layers)
        self.episode_encoder = None
        self.episode_chain = None
        self.episode_cache = None
        if self._uses_lilac:
            lilac = config.context.lilac
            self._lilac_observation_key = str(lilac.observation_key)
            if self._lilac_observation_key not in physical_shapes:
                raise ValueError(f"LILAC input '{self._lilac_observation_key}' is not present in the observation space")
            lilac_shape = physical_shapes[self._lilac_observation_key]
            if len(lilac_shape) != 1:
                raise ValueError('LILAC requires a one-dimensional observation')
            self._lilac_observation_dim = int(lilac_shape[0])
            context_dim = int(lilac.context_dim)
            self._lilac_chain_length = int(lilac.chain_length)
            self.episode_encoder = conditioning.EpisodeSummaryEncoder(self._lilac_observation_dim, self.act_dim, lilac.units, lilac.layers, context_dim)
            self.episode_chain = conditioning.EpisodeChain(context_dim, lilac.chain_units)
            self.episode_cache = conditioning.EpisodeLatentCache(lilac.env_num, lilac.max_episodes, context_dim)
            if self._lilac_chain_length < 1 or self._lilac_chain_length > self.episode_cache.max_episodes:
                raise ValueError('LILAC chain_length must lie between 1 and max_episodes')
        posterior_context_dim = context_dim if self.context_mode in {'dali_s', 'lilac'} else 0
        transition_context_dim = context_dim if self._deep_context else 0
        head_context_dim = context_dim if self._deep_context else 0
        self.context_dim = context_dim
        self.head_context_dim = head_context_dim
        self.rssm = rssm.RSSM(config.rssm, self.embed_size, self.act_dim, transition_context_dim=transition_context_dim, posterior_context_dim=posterior_context_dim, carry_state=self.context_mode == 'carried')
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size + head_context_dim)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size + head_context_dim)
        actor_config = copy.deepcopy(config.actor)
        actor_config.shape = tuple(map(int, act_space.shape))
        self.actor = networks.MLPHead(actor_config, self.rssm.feat_size + head_context_dim)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size + head_context_dim)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0
        self._loss_scales = dict(config.loss_scales)
        if self._uses_dali:
            self._loss_scales['dali_forward'] = float(config.context.dali.loss_scale)
        if self._uses_lilac:
            self._loss_scales['lilac_chain'] = float(config.context.lilac.chain_scale)
            self._loss_scales['lilac_consistency'] = float(config.context.lilac.consistency_scale)
        modules = {'rssm': self.rssm, 'actor': self.actor, 'value': self.value, 'reward': self.reward, 'cont': self.cont, 'encoder': self.encoder}
        context_module_names = set()
        context_lr = None
        if self._uses_dali:
            modules.update({'context_encoder': self.context_encoder, 'forward_dynamics': self.forward_dynamics})
            context_module_names.update({'context_encoder', 'forward_dynamics'})
            context_lr = float(config.context.dali.lr)
        if self._uses_lilac:
            modules.update({'episode_encoder': self.episode_encoder, 'episode_chain': self.episode_chain})
            context_module_names.update({'episode_encoder', 'episode_chain'})
            context_lr = float(config.context.lilac.lr)
        self.decoder = networks.MultiDecoder(config.decoder, self.rssm._deter, self.rssm.flat_stoch, physical_shapes, condition_dim=head_context_dim)
        recon = self._loss_scales.pop('recon')
        self._loss_scales.update({k: recon for k in self.decoder.all_keys})
        modules.update({'decoder': self.decoder})
        self._named_params = OrderedDict()
        for name, module in modules.items():
            for param_name, param in module.named_parameters():
                self._named_params[f'{name}.{param_name}'] = param

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)
        self._agc = _agc
        context_params = [parameter for name, parameter in self._named_params.items() if name.split('.', 1)[0] in context_module_names]
        context_param_ids = {id(parameter) for parameter in context_params}
        base_params = [parameter for parameter in self._named_params.values() if id(parameter) not in context_param_ids]
        optimizer_params = base_params
        if context_params:
            optimizer_params = [{'params': base_params}, {'params': context_params, 'lr': context_lr}]
        self._optimizer = LaProp(optimizer_params, lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps)
        self._scaler = GradScaler(enabled=self.device.type == 'cuda')

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0
        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)
        self.train()
        self.clone_and_freeze()
        if config.compile:
            self._cal_grad = torch.compile(self._cal_grad, mode='reduce-overhead')

    def _update_slow_target(self):
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        self._slow_value.train(False)
        for name in ('encoder', 'rssm', 'reward', 'cont', 'actor', 'value', 'slow_value', 'context_encoder', 'episode_encoder', 'episode_chain'):
            frozen = getattr(self, f'_frozen_{name}', None)
            if frozen is not None:
                frozen.train(False)
        return self

    def clone_and_freeze(self):
        modules = {'encoder': self.encoder, 'rssm': self.rssm, 'reward': self.reward, 'cont': self.cont, 'actor': self.actor, 'value': self.value, 'slow_value': self._slow_value}
        if self._uses_dali:
            modules['context_encoder'] = self.context_encoder
        if self._uses_lilac:
            modules['episode_encoder'] = self.episode_encoder
            modules['episode_chain'] = self.episode_chain
        for name, module in modules.items():
            frozen = copy.deepcopy(module)
            for (source_name, source), (target_name, target) in zip(module.named_parameters(), frozen.named_parameters()):
                assert source_name == target_name
                target.data = source.data
                target.requires_grad_(False)
            frozen.train(False)
            setattr(self, f'_frozen_{name}', frozen)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.clone_and_freeze()
        return self

    def _normalize_context(self, data):
        context = to_f32(data[self.context_key])
        if self._normalize_true_context:
            span = self._context_high_buffer - self._context_low_buffer
            span = torch.where(span > 0, span, torch.ones_like(span))
            context = 2.0 * (context - self._context_low_buffer) / span - 1.0
        return context

    def _dali_observation(self, data):
        return to_f32(data[self._dali_observation_key])

    def _head_feat(self, feat, context):
        if not self._deep_context:
            return feat
        if context is None:
            raise ValueError('deep context conditioning requires context')
        return torch.cat([feat, context], dim=-1)

    def _lilac_observations(self, data):
        observation = to_f32(data[self._lilac_observation_key])
        previous_observation = torch.cat([torch.zeros_like(observation[:, :1]), observation[:, :-1]], dim=1)
        return (observation, previous_observation)

    def _online_lilac_context(self, obs, state, eval=False):
        reset = obs['is_first'].bool()
        if reset.ndim == 2 and reset.shape[-1] == 1:
            reset = reset.squeeze(-1)
        batch = reset.shape[0]
        episode = state['lilac_episode']
        finish = reset & (episode >= 0)
        pooled = state['lilac_sum'] / state['lilac_count'].clamp_min(1.0)
        latent = self._frozen_episode_encoder.summarize(pooled)
        prediction, hidden = self._frozen_episode_chain.step(latent, state['lilac_h'])
        hidden = torch.where(finish[:, None], hidden, state['lilac_h'])
        context = torch.where(finish[:, None], prediction, state['lilac_context'])
        next_episode = episode + reset.to(episode.dtype)
        if not eval:
            if batch > self.episode_cache.env_num:
                raise ValueError('LILAC collection received more environments than the episode cache holds')
            env = torch.arange(batch, device=reset.device)
            self.episode_cache.write_latent(env, episode.clamp_min(0), latent, finish)
            self.episode_cache.write_prediction(env, next_episode, context, reset)
        observation = to_f32(obs[self._lilac_observation_key])
        features = self._frozen_episode_encoder.transition_features(state['lilac_prev_obs'], state['prev_action'], observation)
        keep = ~reset[:, None]
        context_state = {'lilac_prev_obs': observation, 'lilac_sum': torch.where(keep, state['lilac_sum'] + features, torch.zeros_like(features)), 'lilac_count': torch.where(keep, state['lilac_count'] + 1.0, torch.zeros_like(state['lilac_count'])), 'lilac_h': hidden, 'lilac_context': context, 'lilac_episode': next_episode}
        return (context, context_state)

    def _online_dali_context(self, obs, state):
        reset = obs['is_first'].bool()
        if reset.ndim == 2 and reset.shape[-1] == 1:
            reset = reset.squeeze(-1)
        keep = ~reset
        observations = {}
        context_state = {}
        history_length = self._dali_window - 1
        for index, (key, shape) in enumerate(self._dali_history_shapes.items()):
            state_key = f'dali_history_{index}'
            history = state[state_key]
            keep_shape = (keep.shape[0], 1, *(1,) * len(shape))
            history = torch.where(keep.reshape(keep_shape), history, torch.zeros_like(history))
            sequence = torch.cat([history, obs[key][:, None]], dim=1)
            observations[key] = sequence
            context_state[state_key] = sequence[:, -history_length:] if history_length else sequence[:, :0]
        observation = observations[self._dali_observation_key]
        history_action = torch.where(keep[:, None, None], state['dali_action'], torch.zeros_like(state['dali_action']))
        history_valid = state['dali_valid'] & keep[:, None]
        current_action = torch.where(keep[:, None], state['prev_action'], torch.zeros_like(state['prev_action']))
        actions = torch.cat([history_action, current_action[:, None]], dim=1)
        valid = torch.cat([history_valid, torch.ones(reset.shape[0], 1, dtype=torch.bool, device=reset.device)], dim=1)
        observation = torch.where(valid.unsqueeze(-1), observation, torch.zeros_like(observation))
        actions = torch.where(valid.unsqueeze(-1), actions, torch.zeros_like(actions))
        context = self._frozen_context_encoder.encode_window(observation, actions)
        if history_length:
            next_action = actions[:, -history_length:]
            next_valid = valid[:, -history_length:]
        else:
            next_action = actions[:, :0]
            next_valid = valid[:, :0]
        context_state.update({'dali_action': next_action, 'dali_valid': next_valid})
        return (context, context_state)

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        torch.compiler.cudagraph_mark_step_begin()
        embed = self._frozen_encoder(obs)
        context_state = {}
        if self._uses_dali:
            context, context_state = self._online_dali_context(obs, state)
        elif self._uses_lilac:
            context, context_state = self._online_lilac_context(obs, state, eval)
        elif self._uses_true_context:
            context = self._normalize_context(obs)
        else:
            context = None
        prev_stoch, prev_deter, prev_action = (state['stoch'], state['deter'], state['prev_action'])
        stoch, deter, _ = self._frozen_rssm.obs_step(prev_stoch, prev_deter, prev_action, embed, obs['is_first'], context)
        feat = self._frozen_rssm.get_feat(stoch, deter)
        action_dist = self._frozen_actor(self._head_feat(feat, context))
        action = action_dist.mode if eval else action_dist.rsample()
        next_state = {'stoch': stoch, 'deter': deter, 'prev_action': action, **context_state}
        return (action, TensorDict(next_state, batch_size=state.batch_size))

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        state = {'stoch': stoch, 'deter': deter, 'prev_action': action}
        if self._uses_dali:
            history_length = self._dali_window - 1
            history = {f'dali_history_{index}': torch.zeros(B, history_length, *shape, dtype=torch.float32, device=self.device) for index, shape in enumerate(self._dali_history_shapes.values())}
            state.update({**history, 'dali_action': torch.zeros(B, history_length, self.act_dim, dtype=torch.float32, device=self.device), 'dali_valid': torch.zeros(B, history_length, dtype=torch.bool, device=self.device)})
        if self._uses_lilac:
            prediction, hidden = self._frozen_episode_chain.initial(B)
            state.update({'lilac_prev_obs': torch.zeros(B, self._lilac_observation_dim, dtype=torch.float32, device=self.device), 'lilac_sum': torch.zeros(B, self.episode_encoder.units, dtype=torch.float32, device=self.device), 'lilac_count': torch.zeros(B, 1, dtype=torch.float32, device=self.device), 'lilac_h': hidden, 'lilac_context': prediction, 'lilac_episode': torch.full((B,), -1, dtype=torch.int32, device=self.device)})
        return TensorDict(state, batch_size=(B,))

    def update(self, replay_buffer):
        data, index, initial, context_data = replay_buffer.sample()
        torch.compiler.cudagraph_mark_step_begin()
        self._update_slow_target()
        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == 'cuda'):
            stoch, deter = self._cal_grad(data, initial, context_data)
        self._scaler.unscale_(self._optimizer)
        self._agc(self._named_params.values())
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        replay_buffer.update(index, stoch.detach(), deter.detach())

    def _training_context(self, data, context_data):
        if self._uses_true_context:
            return self._normalize_context(data), {}
        if self._uses_lilac:
            return self._lilac_training_context(data)
        if not self._uses_dali:
            return None, {}
        if context_data is None:
            raise ValueError('DALI training requires replay context history and forward targets')
        history_observation = self._dali_observation(context_data['history'])
        target_observation = self._dali_observation(context_data['target'])
        inferred = self.context_encoder(history_observation, context_data['history']['action'], context_data['history']['is_first'])
        steps = data.shape[1]
        context = inferred[:, -steps:]
        current_observation = history_observation[:, -steps:]
        prediction = self.forward_dynamics(current_observation, context_data['forward_action'], context)
        per_step_loss = (prediction - target_observation).square().mean(dim=-1)
        target_reset = context_data['target']['is_first'].bool()
        if target_reset.ndim == 3 and target_reset.shape[-1] == 1:
            target_reset = target_reset.squeeze(-1)
        valid = ~target_reset
        forward_loss = (per_step_loss * valid).sum() / valid.sum().clamp_min(1)
        return context.detach(), {'dali_forward': forward_loss}

    def _lilac_training_context(self, data):
        for key in ('episode', 'episode_index', 'is_last'):
            if key not in data.keys():
                raise ValueError(f"LILAC training requires the replay '{key}' key")
        observation, previous_observation = self._lilac_observations(data)
        reset = data['is_first'].bool()
        if reset.ndim == 3 and reset.shape[-1] == 1:
            reset = reset.squeeze(-1)
        context = self.episode_encoder.encode_segment(previous_observation, data['action'], observation, reset)
        terminal = context[:, -1]
        env = data['episode'][:, -1].long()
        episode = data['episode_index'][:, -1].long()
        single_episode = ~reset[:, 1:].any(dim=1)
        ends_episode = single_episode & data['is_last'][:, -1].reshape(-1).bool()
        current = self.episode_cache.is_current(env, episode)
        self.episode_cache.write_latent(env, episode, terminal.detach(), ends_episode & current)
        latents, valid = self.episode_cache.recent(self._lilac_chain_length)
        prediction = self.episode_chain(latents)
        chain_error = (prediction - latents).square().mean(dim=-1)
        chain_loss = (chain_error * valid).sum() / valid.sum().clamp_min(1)
        target, has_target = self.episode_cache.read_prediction(env, episode)
        row = single_episode & has_target
        consistency_error = (terminal - target.to(terminal.dtype)).square().mean(dim=-1)
        consistency_loss = (consistency_error * row).sum() / row.sum().clamp_min(1)
        losses = {'lilac_chain': chain_loss, 'lilac_consistency': consistency_loss}
        return context, losses

    def _cal_grad(self, data, initial, context_data=None):
        losses = {}
        B, T = data.shape
        embed = self.encoder(data)
        context, context_losses = self._training_context(data, context_data)
        losses.update(context_losses)
        post_stoch, post_deter, post_logit = self.rssm.observe(embed, data['action'], initial, data['is_first'], context)
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses['dyn'] = torch.mean(dyn_loss)
        losses['rep'] = torch.mean(rep_loss)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        decoder_context = context if self._deep_context else None
        recon_losses = {}
        for key, dist in self.decoder(post_stoch, self.rssm.deter_output(post_deter), decoder_context).items():
            recon_losses[key] = torch.mean(-dist.log_prob(data[key]))
        losses.update(recon_losses)
        head_feat = self._head_feat(feat, context)
        losses['rew'] = torch.mean(-self.reward(head_feat).log_prob(to_f32(data['reward'])))
        cont = 1.0 - to_f32(data['is_terminal'])
        losses['con'] = torch.mean(-self.cont(head_feat).log_prob(cont))
        start = (post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(), post_deter.reshape(-1, *post_deter.shape[2:]).detach())
        start_context = None
        if self._deep_context:
            start_context = context.reshape(B * T, -1).detach()
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1, start_context)
        imag_feat, imag_action = (imag_feat.detach(), imag_action.detach())
        imag_reward = self._frozen_reward(imag_feat).mode()
        imag_cont = self._frozen_cont(imag_feat).mean
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)
        _, ret_scale = self.return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale
        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses['policy'] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))
        imag_value_dist = self.value(imag_feat)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses['value'] = torch.mean(weight[:, :-1].detach() * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[:, :-1].unsqueeze(-1))
        last, term, reward = (to_f32(data['is_last']), to_f32(data['is_terminal']), to_f32(data['reward']))
        feat = self._head_feat(self.rssm.get_feat(post_stoch, post_deter), context)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        value_dist = self.value(feat)
        losses['repval'] = torch.mean(weight[:, :-1] * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(-1))
        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()
        return post_stoch, post_deter

    @torch.no_grad()
    def _imagine(self, start, imag_horizon, context=None):
        feats = []
        actions = []
        stoch, deter = start
        for _ in range(imag_horizon):
            feat = self._frozen_rssm.get_feat(stoch, deter)
            feat = self._head_feat(feat, context)
            action = self._frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(action)
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, action, context)
        return (torch.stack(feats, dim=1), torch.stack(actions, dim=1))

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)
