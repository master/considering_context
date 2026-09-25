import gymnasium as gym
import numpy as np

class GymMuJoCo(gym.Env):

    def __init__(self, env, action_repeat=1, seed=0):
        self._env = env
        self._env.reset(seed=seed)
        self._action_repeat = action_repeat
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        shape = self._env.observation_space.shape
        spaces = {'state': gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)}
        if hasattr(self._env, 'context_space'):
            spaces['context'] = self._env.context_space
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        space = self._env.action_space
        return gym.spaces.Box(space.low, space.high, dtype=np.float32)

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0.0
        for _ in range(self._action_repeat):
            state, rew, terminated, truncated, info = self._env.step(action)
            reward += rew
            if terminated or truncated:
                break
        is_last = terminated or truncated
        obs = {'is_first': False, 'is_last': is_last, 'is_terminal': terminated, 'state': state, **self._context()}
        return (obs, reward, is_last, {})

    def reset(self, **kwargs):
        state, info = self._env.reset()
        return {'is_first': True, 'is_last': False, 'is_terminal': False, 'state': state, **self._context()}

    def close(self):
        self._env.close()

    def _context(self):
        if not hasattr(self._env, 'context_space'):
            return {}
        return {'context': np.asarray(self._env.context, dtype=np.float32)}
