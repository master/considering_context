import functools
import warnings
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
CARL_REVISION = 'f4b51b6cfc39b1dea4600750dc52bfae1de50186'

@dataclass(frozen=True)
class TaskSpec:
    environment: str
    task: str
    domain: str
    task_name: str
    context_names: tuple[str, str]
    defaults: tuple[float, float]
    reference_class: str | None
    fixed: tuple[tuple[str, float], ...] = ()
TASK_SPECS = {
    'carl_dmc_walker_walk': TaskSpec('walker', 'carl_dmc_walker_walk', 'walker', 'walk_context', ('gravity', 'actuator_strength'), (9.81, 1.0), 'CARLDmcWalkerEnv'),
    'carl_dmc_walker_walk_wind': TaskSpec('walker', 'carl_dmc_walker_walk_wind', 'walker', 'walk_context', ('actuator_strength', 'wind_x'), (1.0, 0.0), 'CARLDmcWalkerEnv', (('density', 100.0),)),
}

@dataclass(frozen=True)
class DmcContext:
    names: tuple[str, str]
    parameters: tuple[float, float]

    @property
    def vector(self):
        return np.asarray(self.parameters, dtype=np.float32)

    @property
    def values(self):
        return dict(zip(self.names, map(float, self.parameters), strict=True))

class DeterministicContextCycle:

    def __init__(self, contexts, seed, shuffle):
        if not contexts:
            raise ValueError('At least one context is required')
        self._contexts = tuple(contexts)
        self._rng = np.random.default_rng(seed)
        self._shuffle = bool(shuffle)
        self._order = np.arange(len(self._contexts))
        self._position = len(self._order)

    def next(self):
        if self._position == len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._position = 0
        self._position += 1
        return self._contexts[int(self._order[self._position - 1])]

def training_contexts(config):
    spec = task_spec(config)
    low = _context_vector(config.carl.train.low, 'carl.train.low')
    high = _context_vector(config.carl.train.high, 'carl.train.high')
    if np.any(high < low):
        raise ValueError('No CARL training upper bound may fall below its lower bound')
    count = int(config.carl.train.count)
    if count <= 0:
        raise ValueError('carl.train.count must be positive')
    rng = np.random.default_rng(int(config.task_seed))
    values = rng.uniform(low, high, size=(count, 2))
    return tuple(DmcContext(spec.context_names, tuple(map(float, row))) for row in values)

def make(config, env_index):
    _validate_config(config)
    contexts = training_contexts(config)
    selector = DeterministicContextCycle(
        contexts, int(config.seed) + int(env_index), shuffle=True
    )
    return CARLDmc(task_spec(config), selector, int(config.seed) + int(env_index), int(config.action_repeat))

def task_spec(config):
    task = str(config.task)
    if task not in TASK_SPECS:
        raise NotImplementedError(task)
    return TASK_SPECS[task]

@functools.cache
def reference_api(task):
    spec = TASK_SPECS[str(task)]
    with warnings.catch_warnings(record=True):
        import carl.envs.dmc
        from carl.envs.dmc.loader import load_dmc_env
    return (getattr(carl.envs.dmc, spec.reference_class), load_dmc_env)

class CARLDmc(gym.Env):
    metadata = {}

    def __init__(self, spec, selector, seed, action_repeat=1):
        self._spec = spec
        self._selector = selector
        self._seed = int(seed)
        self._action_repeat = int(action_repeat)
        if self._action_repeat <= 0:
            raise ValueError('action_repeat must be positive')
        self._episode = 0
        self._context = self._selector.next()
        self._env = _load_environment(self._spec, self._context, self._seed)
        observation_shape = self._env.observation_spec()['observations'].shape
        action_spec = self._env.action_spec()
        self.observation_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, observation_shape, dtype=np.float32), 'context': gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)})
        self.action_space = gym.spaces.Box(np.asarray(action_spec.minimum, dtype=np.float32), np.asarray(action_spec.maximum, dtype=np.float32), dtype=np.float32)
        self.reward_range = [-np.inf, np.inf]

    @property
    def context(self):
        return self._context.vector

    @property
    def physics(self):
        return self._env.physics

    def reset(self, **kwargs):
        if self._episode:
            self._context = self._selector.next()
            self._env = _load_environment(self._spec, self._context, self._seed + self._episode)
        time_step = self._env.reset()
        self._episode += 1
        return self._observation(time_step)

    def step(self, action):
        if not np.isfinite(action).all():
            raise ValueError(f'CARL {self._spec.environment} actions must be finite')
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += float(time_step.reward or 0.0)
            if time_step.last():
                break
        observation = self._observation(time_step)
        done = bool(time_step.last())
        info = {'discount': np.asarray(time_step.discount, dtype=np.float32)}
        return (observation, reward, done, info)

    def close(self):
        self._env = None

    def _observation(self, time_step):
        discount = time_step.discount
        return {
            'state': np.asarray(time_step.observation['observations'], dtype=np.float32),
            'context': self._context.vector,
            'is_first': bool(time_step.first()),
            'is_last': bool(time_step.last()),
            'is_terminal': False if time_step.first() else discount == 0,
        }

def _load_environment(spec, context, seed):
    environment_class, loader = reference_api(spec.task)
    with warnings.catch_warnings(record=True):
        full_context = environment_class.get_default_context()
    full_context.update(dict(spec.fixed))
    full_context.update(context.values)
    return loader(domain_name=spec.domain, task_name=spec.task_name, context=full_context, task_kwargs={'random': int(seed)}, environment_kwargs={'flat_observation': True})

def _validate_config(config):
    spec = task_spec(config)
    if str(config.carl.environment) != spec.environment:
        raise ValueError(f'carl.environment must be {spec.environment}')
    if tuple(config.carl.context_names) != spec.context_names:
        raise ValueError(f'{spec.environment} context order must be {spec.context_names}')
    defaults = tuple((float(value) for value in config.carl.defaults))
    if defaults != spec.defaults:
        raise ValueError(f'{spec.environment} context defaults must be {spec.defaults}')
    fixed = tuple(sorted(((str(name), float(value)) for name, value in dict(config.carl.get('fixed') or {}).items())))
    if fixed != tuple(sorted(spec.fixed)):
        raise ValueError(f'{config.task} carl.fixed must be {dict(spec.fixed)}')
    if str(config.carl.revision) != CARL_REVISION:
        raise ValueError(f'CARL revision must be {CARL_REVISION}')

def _context_vector(values, name):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (2,) or not np.isfinite(vector).all():
        raise ValueError(f'{name} must contain two finite values')
    return vector
