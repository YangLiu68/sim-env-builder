"""Gymnasium-style wrapper for the scene (template: adapt observation/action/reward to the task).

    from env import SimEnv
    env = SimEnv(render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    frame = env.render()                      # RGB from CAMERA (or a free overview camera)
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces

    _Base = gym.Env
except ImportError:  # works without gymnasium; spaces become simple dicts
    gym, spaces, _Base = None, None, object

HERE = Path(__file__).resolve().parent
SCENE = HERE / "scene.xml"
CAMERA = None          # e.g. "chase" or a robot camera; None = free overview camera
FRAME_SKIP = 10        # physics steps per env step (control at 1 / (timestep * FRAME_SKIP) Hz)
MAX_STEPS = 1000


class SimEnv(_Base):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, width: int = 640, height: int = 360):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        sys.path.insert(0, str(HERE))
        from simkit.world import load_world  # world.py: buoyancy, conveyors, NPCs... (optional)

        self.world = load_world(HERE, self.model)
        self.render_mode, self.width, self.height = render_mode, width, height
        self._renderer = None
        lo = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        hi = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        nobs = len(self._obs())
        if spaces is not None:
            self.action_space = spaces.Box(lo, hi, dtype=np.float32)
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(nobs,), dtype=np.float64)
        self.steps = 0

    # -- task-specific pieces: EDIT THESE ------------------------------------------------
    def _obs(self) -> np.ndarray:
        d = self.data
        return np.concatenate([d.qpos, d.qvel, d.sensordata])

    def _reward(self) -> float:
        return 0.0

    def _terminated(self) -> bool:
        return not np.isfinite(self.data.qpos).all()

    # -- standard API ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if gym is not None:
            super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        _ = rng  # randomise initial state / object poses here if the task needs it
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float64)
        lo, hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(a, lo, hi)
        for _ in range(FRAME_SKIP):
            if self.world is not None:
                self.world(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        obs = self._obs()
        return obs, self._reward(), self._terminated(), self.steps >= MAX_STEPS, {}

    def render(self):
        if self._renderer is None:
            self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.width)
            self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.height)
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        cam = CAMERA if CAMERA is not None else -1
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    env = SimEnv()
    obs, _ = env.reset(seed=0)
    total = 0.0
    for _ in range(50):
        act = np.zeros(env.model.nu) if spaces is None else env.action_space.sample()
        obs, r, term, trunc, _ = env.step(act)
        total += r
        if term or trunc:
            break
    print("obs dim", obs.shape, "return", total, "finite", np.isfinite(obs).all())
