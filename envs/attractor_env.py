from __future__ import annotations
from pathlib import Path
from typing import Any
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
THETA_VALUES = np.array([[-1.5, -1.5], [+1.5, -1.5], [-1.5, +1.5], [+1.5, +1.5]], dtype=np.float64)
WORKSPACE_HALF = 2.5
REFLECTION_DAMPING = 0.5
DEFAULT_K = 0.3
DEFAULT_SIGMA = 0.1
DEFAULT_T = 50
DEFAULT_DT = 0.05

class PointMassAttractorEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, k: float=DEFAULT_K, sigma: float=DEFAULT_SIGMA, max_steps: int=DEFAULT_T, workspace_half: float=WORKSPACE_HALF, reflection_damping: float=REFLECTION_DAMPING, seed: int | None=None):
        self.k = float(k)
        self.sigma = float(sigma)
        self.max_steps = int(max_steps)
        self.workspace_half = float(workspace_half)
        self.reflection_damping = float(reflection_damping)
        xml_path = Path(__file__).with_name('attractor.xml')
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        assert abs(self._model.opt.timestep - DEFAULT_DT) < 1e-09, f'XML timestep {self._model.opt.timestep} != {DEFAULT_DT}'
        self.observation_space = spaces.Box(low=np.array([-self.workspace_half, -self.workspace_half, -np.inf, -np.inf], dtype=np.float32), high=np.array([self.workspace_half, self.workspace_half, np.inf, np.inf], dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._theta_idx: int = 0
        self._theta: np.ndarray = THETA_VALUES[0].copy()
        self._step_count = 0
        self._np_random: np.random.Generator | None = None
        if seed is not None:
            self.reset(seed=seed)

    def _obs(self) -> np.ndarray:
        return np.asarray(np.concatenate([self._data.qpos[:2], self._data.qvel[:2]]), dtype=np.float32)

    def reset(self, *, seed: int | None=None, options: dict[str, Any] | None=None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None or self._np_random is None:
            self._np_random = np.random.default_rng(seed)
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[:2] = self._np_random.uniform(-0.5, 0.5, size=2)
        self._data.qvel[:2] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._theta_idx = int(self._np_random.integers(0, 4))
        self._theta = THETA_VALUES[self._theta_idx].copy()
        self._step_count = 0
        return (self._obs(), self._info())

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        self._data.ctrl[:2] = action
        self._data.qfrc_applied[:2] = -self.k * (self._data.qpos[:2] - self._theta)
        mujoco.mj_step(self._model, self._data)
        dt = self._model.opt.timestep
        xi = self._np_random.standard_normal(size=(2, 2))
        self._data.qvel[:2] += self.sigma * np.sqrt(dt) * xi[0]
        self._data.qpos[:2] += self.sigma * dt ** 1.5 * xi[1]
        for axis in range(2):
            p = self._data.qpos[axis]
            if p > self.workspace_half:
                self._data.qpos[axis] = 2 * self.workspace_half - p
                self._data.qvel[axis] = -self.reflection_damping * self._data.qvel[axis]
            elif p < -self.workspace_half:
                self._data.qpos[axis] = -2 * self.workspace_half - p
                self._data.qvel[axis] = -self.reflection_damping * self._data.qvel[axis]
        self._step_count += 1
        terminated = False
        truncated = self._step_count >= self.max_steps
        obs = self._obs()
        return (obs, 0.0, terminated, truncated, self._info())

    def _info(self) -> dict[str, Any]:
        return {'theta_idx': int(self._theta_idx), 'theta': self._theta.astype(np.float32).copy()}

    def close(self) -> None:
        pass
