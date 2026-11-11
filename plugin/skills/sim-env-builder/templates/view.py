"""Watch the environment live in MuJoCo's interactive viewer WITH the demo controller running
(lights change, pedestrians walk, the robot does its demo). Plain `python -m mujoco.viewer`
only shows passive physics, because scripted behaviour lives in controller.py.

    python view.py                 # demo controller on
    python view.py --passive       # physics only
    (macOS: run with `mjpython view.py` if the viewer asks for it)
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--passive", action="store_true")
ap.add_argument("--speed", type=float, default=1.0)
a = ap.parse_args()

model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
data = mujoco.MjData(model)
if model.nkey:
    mujoco.mj_resetDataKeyframe(model, data, 0)
ctrl = None
if not a.passive and (HERE / "controller.py").exists():
    spec = importlib.util.spec_from_file_location("controller", HERE / "controller.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctrl = mod.make_controller(model) if hasattr(mod, "make_controller") else getattr(mod, "controller", None)

from simkit.world import load_world  # noqa: E402

world = load_world(HERE, model)  # environment dynamics run even with --passive

with mujoco.viewer.launch_passive(model, data) as v:
    t0, s0 = time.perf_counter(), data.time
    while v.is_running():
        target = s0 + (time.perf_counter() - t0) * a.speed
        while data.time < target:
            if world is not None:
                world(model, data)
            if ctrl is not None:
                ctrl(model, data)
            mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.005)
