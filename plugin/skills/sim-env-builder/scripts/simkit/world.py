"""Environment dynamics that are part of the WORLD, not of any policy.

Buoyancy and water drag, wind gusts, conveyor belts moving boxes, traffic lights cycling, scripted
pedestrians/NPC cars: these must run every physics step no matter who drives the robot (demo
controller, RL policy, passive physics test, interactive viewer). Put them in ``<env>/world.py``:

    # world.py
    def make_world(model):
        state = {...}                       # precompute ids, parameters
        def world(model, data):             # called before EVERY mj_step
            data.xfrc_applied[boat] = buoyancy_and_drag(...)
            data.mocap_pos[ped] = ...
        return world

simkit (smoke, physics examiner, showcase), env.py, view.py and the dashboard viewer all call it.
In your own loops use ``simkit.world.step(model, data, world)`` or call ``world(model, data)``
yourself right before ``mujoco.mj_step``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mujoco


def load_world(env_dir, model):
    """The env's world callable from ``world.py`` (make_world(model) or world(model, data)), or None."""
    path = Path(env_dir) / "world.py"
    if not path.exists():
        return None
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"world_{abs(hash(str(path)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "make_world"):
        return mod.make_world(model)
    return getattr(mod, "world", None)


def with_world(world, controller):
    """One callable doing world dynamics then the controller (either may be None)."""
    if world is None:
        return controller
    if controller is None:
        return world

    def both(model, data):
        world(model, data)
        controller(model, data)
    return both


def step(model, data, world=None, n: int = 1):
    for _ in range(n):
        if world is not None:
            world(model, data)
        mujoco.mj_step(model, data)
