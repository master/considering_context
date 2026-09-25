import gymnasium as gym
import numpy as np
P_STAY = 0.9
RHO = 0.9
OMEGA = np.pi / 4
SIGMA_D = 0.3
LAM = 0.9
BETA = 0.5
ACTION_COST = 0.01

def rotation(angle):
    cos, sin = (np.cos(angle), np.sin(angle))
    return np.array([[cos, -sin], [sin, cos]])

class DPMDP(gym.Env):

    def __init__(self, p_stay=P_STAY, rho=RHO, omega=OMEGA, sigma_d=SIGMA_D, lam=LAM, beta=BETA):
        self.p_stay = float(p_stay)
        self.rho = float(rho)
        self.omega = float(omega)
        self.sigma_d = float(sigma_d)
        self.lam = float(lam)
        self.beta = float(beta)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.context_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self._rng = None
        self.z = 1
        self._x = np.zeros(2)
        self._v = np.zeros(2)
        self._d = np.zeros(2)

    @property
    def context(self):
        return np.asarray([self.z], np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
            self.z = 1 if self._rng.random() < 0.5 else -1
        elif self._rng.random() >= self.p_stay:
            self.z = -self.z
        self._x = np.zeros(2)
        self._v = np.zeros(2)
        self._d = self.sigma_d * self._rng.standard_normal(2)
        return (self._observation(), {})

    def step(self, action):
        action = np.clip(np.asarray(action, np.float64), self.action_space.low, self.action_space.high)
        noise = self.sigma_d * np.sqrt(1.0 - self.rho ** 2) * self._rng.standard_normal(2)
        self._d = self.rho * rotation(self.z * self.omega) @ self._d + noise
        self._v = self.lam * self._v + self.beta * action + self._d
        self._x = self._x + self._v
        reward = -float(self._x @ self._x + ACTION_COST * action @ action)
        return (self._observation(), reward, False, False, {})

    def _observation(self):
        return np.concatenate([self._x, self._v, self._d]).astype(np.float32)

def make(task, **kwargs):
    return DPMDP(**kwargs)
