"""Render a showcase MP4 of an environment: orbit fly-around, then the demo controller running
(tracking camera on the robot), then each named camera. Titles are burned in.

    python -m simkit.showcase <env_dir> [--out showcase.mp4] [--seconds 10] [--controller controller.py]
    (simkit.smoke writes smoke/showcase.mp4 automatically)
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .render import actuated_roots, body_label, free_camera, is_mobile, pull_in_front, scene_bounds, subtree_bounds, unblock_tracking

W, H, FPS = 1280, 720, 30


def font(n):
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(p, n)
        except OSError:
            pass
    return ImageFont.load_default()


def title(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    d.rounded_rectangle([18, 18, 30 + max(len(text) * 15, len(sub) * 10), 82 if sub else 58], 10, fill=(10, 13, 19, 170))
    d.text((32, 24), text, fill=(240, 244, 250), font=font(24))
    if sub:
        d.text((32, 56), sub, fill=(170, 180, 200), font=font(17))
    return np.asarray(im)


def load_controller(env: Path, path: Path | None, model):
    path = path or env / "controller.py"
    if not path.exists():
        return None
    sys.path.insert(0, str(env))
    spec = importlib.util.spec_from_file_location("showcase_ctrl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_controller(model) if hasattr(mod, "make_controller") else getattr(mod, "controller", None)


def reset(m, d):
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)


def render(env_dir, out=None, seconds=10.0, controller=None, scene="scene.xml", name=None, orbit_s=4.0, cam_s=2.5):
    env = Path(env_dir).resolve()
    m = mujoco.MjModel.from_xml_path(str(env / scene))
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, W), max(m.vis.global_.offheight, H)
    d = mujoco.MjData(m)
    reset(m, d)
    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    for i in range(6):
        opt.geomgroup[i] = 1 if i <= 2 else 0
    frames = []
    label = name or env.name
    lo, hi = scene_bounds(m, d)
    c = (lo + hi) / 2
    span = float(np.linalg.norm(hi - lo))
    dist = max(span * 0.62, 1.5)
    # 1) orbit
    n = int(orbit_s * FPS)
    for k in range(n):
        cam = pull_in_front(m, d, free_camera(c, dist, 120 + 70 * k / n, -24 + 6 * math.sin(math.pi * k / n)))
        r.update_scene(d, camera=cam, scene_option=opt)
        frames.append(title(r.render(), label, "scene overview"))
    # 2) controller rollout with tracking camera
    from .world import load_world, with_world

    ctrl = with_world(load_world(env, m), load_controller(env, Path(controller) if controller else None, m))
    roots = actuated_roots(m)
    fixed_base = bool(roots) and not is_mobile(m, roots[0])
    if fixed_base:  # arms etc.: slow orbit around the workspace instead of a camera inside the links
        wlo, whi = subtree_bounds(m, d, roots[0])
        wc, wspan = (wlo + whi) / 2, max(float(np.linalg.norm(whi - wlo)), 0.5)
        roots = []
    if roots:
        tr = mujoco.MjvCamera()
        tr.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        tr.trackbodyid = roots[0]
        blo, bhi = subtree_bounds(m, d, roots[0])
        tr.distance = max(float(np.linalg.norm(bhi - blo)) * 3.0, 1.6)
        tr.elevation = -18
        track_dist = tr.distance
    every = max(1, int(round(1 / (FPS * m.opt.timestep))))
    steps = int(seconds / m.opt.timestep)
    who = body_label(m, roots[0]) if roots else ""
    for k in range(steps):
        if ctrl is not None:
            ctrl(m, d)
        mujoco.mj_step(m, d)
        if k % every == 0:
            if roots:
                tr.azimuth = 135 + 25 * math.sin(d.time * 0.3)
                unblock_tracking(m, d, tr, nominal=track_dist)
                r.update_scene(d, camera=tr, scene_option=opt)
            elif fixed_base:
                r.update_scene(d, camera=pull_in_front(m, d, free_camera(wc, wspan * 1.5, 120 + 4 * d.time, -26)), scene_option=opt)
            else:
                r.update_scene(d, camera=pull_in_front(m, d, free_camera(c, dist, 190 + 10 * d.time, -22)), scene_option=opt)
            frames.append(title(r.render(), label, f"{'demo controller' if ctrl else 'passive physics'}{' · following ' + who if who else (' · robot workspace' if fixed_base else '')} · t = {d.time:4.1f} s"))
    # 3) named cameras, continuing the rollout
    for cid in range(m.ncam):
        cname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, cid) or f"camera {cid}"
        for k in range(int(cam_s * FPS)):
            for _ in range(every):
                if ctrl is not None:
                    ctrl(m, d)
                mujoco.mj_step(m, d)
            r.update_scene(d, camera=cid, scene_option=opt)
            frames.append(title(r.render(), label, f"camera: {cname} · t = {d.time:4.1f} s"))
    r.close()
    out = Path(out or env / "showcase.mp4")
    import imageio.v2 as iio

    iio.mimwrite(out, frames, fps=FPS, quality=6, macro_block_size=1)
    return out, len(frames) / FPS


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--out")
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--controller")
    ap.add_argument("--scene", default="scene.xml")
    ap.add_argument("--name")
    a = ap.parse_args()
    p, dur = render(a.env, a.out, a.seconds, a.controller, a.scene, a.name)
    print(f"wrote {p} ({dur:.1f} s)")
