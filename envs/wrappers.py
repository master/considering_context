import gymnasium as gym
import numpy as np
import tools

class TimeLimit(gym.Wrapper):

    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if 'discount' not in info:
                info['discount'] = np.array(1.0).astype(np.float32)
            self._step = None
            obs['is_last'] = True
        return (obs, reward, done, info)

    def reset(self):
        self._step = 0
        return self.env.reset()

class NormalizeActions(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(np.isfinite(env.action_space.low), np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)

class Dtype(gym.Wrapper):

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return (tools.convert(obs), np.float32(rew), done, info)

    def reset(self):
        return tools.convert(self.env.reset())
