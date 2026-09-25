from xml.etree import ElementTree
import gymnasium as gym
import numpy as np
TASK = 'rwrl_walker_walk'
PARAMETERS = ('joint_damping',)

class RandomWalkChain:

    def __init__(self, low, high, std, rng):
        self.low = float(low)
        self.high = float(high)
        self.std = float(std)
        if not np.isfinite([self.low, self.high]).all() or self.high <= self.low:
            raise ValueError('RWRL bounds must be finite with high above low')
        if not self.std > 0:
            raise ValueError('RWRL step size must be positive')
        self._rng = rng
        self._value = None

    def next(self):
        if self._value is None:
            self._value = float(self._rng.uniform(self.low, self.high))
        else:
            self._value = self._fold(self._value + self.std * float(self._rng.standard_normal()))
        return self._value

    def _fold(self, value):
        width = self.high - self.low
        offset = (value - self.low) % (2.0 * width)
        return self.low + (2.0 * width - offset if offset > width else offset)

def chain(config, env_index, train):
    if str(config.task) != TASK:
        raise NotImplementedError(str(config.task))
    settings = config.rwrl
    rate = str(settings.rate)
    if rate not in settings.rates:
        raise ValueError(f'RWRL rate must be one of {sorted(settings.rates)}, got {rate!r}')
    seed = int(config.seed) if train else int(settings.eval_chain_seed)
    rng = np.random.default_rng([seed, int(env_index)])
    return RandomWalkChain(settings.low, settings.high, settings.rates[rate], rng)

def make(config, env_index, train):
    return RWRLWalker(str(config.rwrl.parameter), chain(config, env_index, train), int(config.seed) + int(env_index), int(config.action_repeat))

class RWRLWalker(gym.Env):
    metadata = {}

    def __init__(self, parameter, chain, seed, action_repeat=1):
        if parameter not in PARAMETERS:
            raise ValueError(f'RWRL parameter must be one of {PARAMETERS}, got {parameter!r}')
        self._parameter = str(parameter)
        self._chain = chain
        self._seed = int(seed)
        self._action_repeat = int(action_repeat)
        if self._action_repeat <= 0:
            raise ValueError('action_repeat must be positive')
        self._episode = 0
        self._theta = self._chain.next()
        self._env = _load_environment(self._parameter, self._theta, self._seed)
        state_shape = self._env.observation_spec()['observations'].shape
        action_spec = self._env.action_spec()
        self.observation_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, state_shape, dtype=np.float32), 'context': gym.spaces.Box(chain.low, chain.high, (1,), dtype=np.float32)})
        self.action_space = gym.spaces.Box(np.asarray(action_spec.minimum, dtype=np.float32), np.asarray(action_spec.maximum, dtype=np.float32), dtype=np.float32)
        self.reward_range = [-np.inf, np.inf]

    @property
    def context(self):
        return np.asarray([self._theta], dtype=np.float32)

    @property
    def physics(self):
        return self._env.physics

    def reset(self, **kwargs):
        if self._episode:
            self._theta = self._chain.next()
            self._env = _load_environment(self._parameter, self._theta, self._seed + self._episode)
        time_step = self._env.reset()
        self._episode += 1
        return self._observation(time_step)

    def step(self, action):
        if not np.isfinite(action).all():
            raise ValueError('RWRL walker actions must be finite')
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            reward += float(time_step.reward or 0.0)
            if time_step.last():
                break
        observation = self._observation(time_step)
        info = {'discount': np.asarray(time_step.discount, dtype=np.float32)}
        return (observation, reward, bool(time_step.last()), info)

    def close(self):
        self._env = None

    def _observation(self, time_step):
        return {
            'state': np.asarray(time_step.observation['observations'], dtype=np.float32),
            'context': self.context,
            'is_first': bool(time_step.first()),
            'is_last': bool(time_step.last()),
            'is_terminal': False if time_step.first() else bool(time_step.discount == 0),
        }

def _load_environment(parameter, theta, seed):
    from dm_control.rl import control
    from dm_control.suite import walker
    xml, assets = walker.get_model_and_assets()
    root = ElementTree.fromstring(xml)
    _edit_model(root, parameter, float(theta))
    physics = walker.Physics.from_xml_string(ElementTree.tostring(root), assets)
    task = walker.PlanarWalker(move_speed=walker._WALK_SPEED, random=int(seed))
    return control.Environment(physics, task, time_limit=walker._DEFAULT_TIME_LIMIT, control_timestep=walker._CONTROL_TIMESTEP, flat_observation=True)

def _edit_model(root, parameter, theta):
    if parameter != 'joint_damping':
        raise ValueError(parameter)
    root.find('./default/joint').set('damping', str(theta))
