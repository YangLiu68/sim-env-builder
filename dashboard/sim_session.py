"""Live MuJoCo sessions for the browser viewer.

``export_scene(model)`` turns a compiled model into JSON that three.js can draw (geoms per body,
meshes, heightfields, textures for planes, cameras, actuators). ``SimSession`` steps the physics
in real time on a thread and hands out compact pose frames (body xpos/xquat + camera poses).
"""

from __future__ import annotations

import base64
import importlib.util
import io
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

GEOM_TYPES = {int(mujoco.mjtGeom.mjGEOM_PLANE): "plane", int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
              int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
              int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid", int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
              int(mujoco.mjtGeom.mjGEOM_BOX): "box", int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def _texture_png(model, tex_id: int) -> str | None:
    try:
        from PIL import Image

        w, h, c = int(model.tex_width[tex_id]), int(model.tex_height[tex_id]), int(model.tex_nchannel[tex_id])
        adr = int(model.tex_adr[tex_id])
        arr = np.asarray(model.tex_data[adr: adr + w * h * c], np.uint8).reshape(h, w, c)
        im = Image.fromarray(arr[..., :3] if c >= 3 else arr[..., 0])
        if max(w, h) > 512:
            im.thumbnail((512, 512))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _rgb_texid(model, mat: int) -> int:
    t = model.mat_texid[mat]
    if np.ndim(t) == 0:
        return int(t)
    role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB) if hasattr(mujoco, "mjtTextureRole") else 1
    return int(t[role]) if t[role] >= 0 else int(max(t))


def export_scene(model) -> dict:
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    name = lambda obj, i: mujoco.mj_id2name(model, obj, i) or ""  # noqa: E731
    geoms, meshes, hfields, textures = [], {}, {}, {}
    for g in range(model.ngeom):
        t = GEOM_TYPES.get(int(model.geom_type[g]))
        if t is None:
            continue
        mat = int(model.geom_matid[g])
        own = model.geom_rgba[g]
        use_own = mat < 0 or not np.allclose(own, [0.5, 0.5, 0.5, 1.0])
        rgba = (own if use_own else model.mat_rgba[mat]).tolist()
        item = {"id": g, "name": name(mujoco.mjtObj.mjOBJ_GEOM, g), "body": int(model.geom_bodyid[g]), "type": t,
                "size": model.geom_size[g].tolist(), "pos": model.geom_pos[g].tolist(), "quat": model.geom_quat[g].tolist(),
                "rgba": rgba, "group": int(model.geom_group[g]),
                "collide": bool(model.geom_contype[g] or model.geom_conaffinity[g])}
        if mat >= 0:
            tid = _rgb_texid(model, mat)
            if tid >= 0 and int(model.tex_type[tid]) == int(mujoco.mjtTexture.mjTEXTURE_2D):
                if tid not in textures:
                    textures[tid] = _texture_png(model, tid)
                item["texture"] = tid
                item["texrepeat"] = model.mat_texrepeat[mat].tolist()
                item["texuniform"] = bool(model.mat_texuniform[mat])
            item["reflectance"] = float(model.mat_reflectance[mat])
        if t == "mesh":
            mid = int(model.geom_dataid[g])
            item["mesh"] = mid
            if mid not in meshes:
                va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                meshes[mid] = {"vert": _b64(model.mesh_vert[va:va + vn], np.float32),
                               "face": _b64(model.mesh_face[fa:fa + fn], np.int32), "nv": int(vn), "nf": int(fn)}
        if t == "hfield":
            hid = int(model.geom_dataid[g])
            item["hfield"] = hid
            if hid not in hfields:
                nr, nc = int(model.hfield_nrow[hid]), int(model.hfield_ncol[hid])
                adr = int(model.hfield_adr[hid])
                hfields[hid] = {"nrow": nr, "ncol": nc, "size": model.hfield_size[hid].tolist(),
                                "data": _b64(model.hfield_data[adr: adr + nr * nc], np.float32)}
        geoms.append(item)
    bodies = [{"id": b, "name": name(mujoco.mjtObj.mjOBJ_BODY, b), "parent": int(model.body_parentid[b]),
               "mocap": bool(model.body_mocapid[b] >= 0)} for b in range(model.nbody)]
    cams = [{"id": c, "name": name(mujoco.mjtObj.mjOBJ_CAMERA, c) or f"camera {c}", "fovy": float(model.cam_fovy[c]),
             "body": int(model.cam_bodyid[c])} for c in range(model.ncam)]
    acts = []
    for a in range(model.nu):
        lim = bool(model.actuator_ctrllimited[a])
        rng = model.actuator_ctrlrange[a].tolist() if lim else [-1.0, 1.0]
        acts.append({"id": a, "name": name(mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act{a}", "range": rng, "limited": lim,
                     "default": float(data.ctrl[a])})
    lo, hi = _bounds(model, data)
    sky = _sky_colors(model)
    return {"bodies": bodies, "geoms": geoms, "meshes": {str(k): v for k, v in meshes.items()},
            "hfields": {str(k): v for k, v in hfields.items()}, "textures": {str(k): v for k, v in textures.items() if v},
            "cameras": cams, "actuators": acts, "bounds": [lo.tolist(), hi.tolist()], "timestep": model.opt.timestep,
            "gravity": model.opt.gravity.tolist(), "sky": sky, "nbody": model.nbody,
            "keyframes": [name(mujoco.mjtObj.mjOBJ_KEY, k) or f"key{k}" for k in range(model.nkey)]}


def _bounds(model, data):
    pts = []
    for g in range(model.ngeom):
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        R = data.geom_xmat[g].reshape(3, 3)
        ctr = data.geom_xpos[g] + R @ c
        half = np.abs(R) @ h
        pts += [ctr - half, ctr + half]
    if not pts:
        return np.array([-2.0, -2, 0]), np.array([2.0, 2, 2])
    p = np.array(pts)
    if len(p) > 40:
        return np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    return p.min(0), p.max(0)


def _sky_colors(model):
    for t in range(model.ntex):
        if int(model.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
            w, h, c = int(model.tex_width[t]), int(model.tex_height[t]), int(model.tex_nchannel[t])
            adr = int(model.tex_adr[t])
            arr = np.asarray(model.tex_data[adr: adr + w * h * c], np.uint8).reshape(h, w, c)
            # skybox textures stack 6 faces vertically: +x -x +y -y +z -z (top face ~ 4th sixth)
            face = h // 6 if h >= 6 * w // 2 else h
            top = arr[face * 4: face * 5].reshape(-1, c)[:, :3].mean(0) if h >= 6 * w // 2 else arr[: h // 4].reshape(-1, c)[:, :3].mean(0)
            side = arr[face * 0: face * 1].reshape(-1, c)[:, :3]
            horizon = side[len(side) // 2: len(side) // 2 + w].mean(0)
            return {"top": (top / 255).tolist(), "horizon": (horizon / 255).tolist()}
    return {"top": [0.55, 0.7, 0.9], "horizon": [0.9, 0.93, 0.97]}


def load_controller(env_dir: Path, model):
    """The environment's own scripted demo controller (controller.py), or None."""
    path = env_dir / "controller.py"
    if not path.exists():
        return None
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    spec = importlib.util.spec_from_file_location(f"ctrl_{abs(hash(str(env_dir)))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "make_controller"):
        return mod.make_controller(model)
    return getattr(mod, "controller", None)


class SimSession:
    """Real-time stepping thread. ``frame()`` returns the latest pose frame as float32 bytes:
    [time, rtf, ncon, nbody*3 xpos, nbody*4 xquat, ncam*3 cam pos, ncam*9 cam mat]."""

    def __init__(self, env_dir: Path, scene: str = "scene.xml"):
        self.env_dir = Path(env_dir)
        self.model = mujoco.MjModel.from_xml_path(str(self.env_dir / scene))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.playing = True
        self.use_demo = True
        self.speed = 1.0
        self.manual = None
        self.controller = None
        self.controller_error = None
        self._reset()
        self._stop = False
        self._rtf = 1.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _reset(self):
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.manual = self.data.ctrl.copy()
        try:
            self.controller = load_controller(self.env_dir, self.model)
            self.controller_error = None
        except Exception as e:  # noqa: BLE001
            self.controller, self.controller_error = None, repr(e)
        self.world = None
        try:  # world.py dynamics (buoyancy, conveyors, NPCs) always run, demo or manual
            if (self.env_dir / "world.py").exists():
                sys.path.insert(0, str(self.env_dir))
                spec = importlib.util.spec_from_file_location(f"world_{abs(hash(str(self.env_dir)))}", self.env_dir / "world.py")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.world = mod.make_world(self.model) if hasattr(mod, "make_world") else getattr(mod, "world", None)
        except Exception as e:  # noqa: BLE001
            self.controller_error = f"world.py: {e!r}"

    def command(self, msg: dict):
        with self.lock:
            cmd = msg.get("cmd")
            if cmd == "play":
                self.playing = True
            elif cmd == "pause":
                self.playing = False
            elif cmd == "reset":
                self._reset()
            elif cmd == "step":
                self._step_once()
            elif cmd == "demo":
                self.use_demo = bool(msg.get("on"))
                if not self.use_demo:
                    self.manual = self.data.ctrl.copy()
            elif cmd == "speed":
                self.speed = float(np.clip(msg.get("value", 1.0), 0.05, 8.0))
            elif cmd == "ctrl":
                i, v = int(msg["index"]), float(msg["value"])
                if 0 <= i < self.model.nu:
                    self.manual[i] = v
                    self.use_demo = False
            elif cmd == "push":  # impulse on a body: {"body": id, "force": [fx, fy, fz]}
                b = int(msg.get("body", 0))
                if 0 < b < self.model.nbody:
                    self.data.xfrc_applied[b, :3] = np.asarray(msg.get("force", [0, 0, 0]), float)
                    self._push_until = self.data.time + 0.15

    def _step_once(self):
        if getattr(self, "world", None) is not None:
            try:
                self.world(self.model, self.data)
            except Exception as e:  # noqa: BLE001
                self.controller_error, self.world = f"world.py: {e!r}", None
        if self.use_demo and self.controller is not None:
            try:
                self.controller(self.model, self.data)
            except Exception as e:  # noqa: BLE001
                self.controller_error = repr(e)
                self.use_demo = False
        else:
            self.data.ctrl[:] = self.manual
        mujoco.mj_step(self.model, self.data)
        if getattr(self, "_push_until", None) is not None and self.data.time > self._push_until:
            self.data.xfrc_applied[:] = 0
            self._push_until = None
        if not np.isfinite(self.data.qpos).all():
            self._reset()

    def _loop(self):
        dt = self.model.opt.timestep
        wall0, sim0 = time.perf_counter(), self.data.time
        while not self._stop:
            if not self.playing:
                time.sleep(0.02)
                wall0, sim0 = time.perf_counter(), self.data.time
                continue
            target = sim0 + (time.perf_counter() - wall0) * self.speed
            n = 0
            t_start = time.perf_counter()
            with self.lock:
                while self.data.time < target and n < 2000:
                    self._step_once()
                    n += 1
                if self.data.time < sim0:  # reset happened
                    wall0, sim0 = time.perf_counter(), self.data.time
            spent = time.perf_counter() - t_start
            if n:
                self._rtf = 0.9 * self._rtf + 0.1 * min((n * dt) / max(spent, 1e-6), 999)
            if self.data.time < target - 0.25:  # cannot keep up: drop the backlog
                wall0, sim0 = time.perf_counter(), self.data.time
            time.sleep(0.004)

    def frame(self) -> bytes:
        with self.lock:
            d = self.data
            parts = [np.array([d.time, self._rtf, d.ncon], np.float32), d.xpos.astype(np.float32).ravel(),
                     d.xquat.astype(np.float32).ravel(), d.cam_xpos.astype(np.float32).ravel(), d.cam_xmat.astype(np.float32).ravel()]
        return np.concatenate(parts).tobytes()

    def info(self) -> dict:
        with self.lock:
            return {"time": float(self.data.time), "playing": self.playing, "demo": self.use_demo,
                    "has_controller": self.controller is not None, "controller_error": self.controller_error,
                    "ctrl": self.data.ctrl.tolist(), "speed": self.speed}

    def render_png(self, camera: str | int = -1, width=960, height=540) -> bytes:
        from PIL import Image

        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        with self.lock:
            r = mujoco.Renderer(self.model, height, width)
            try:
                r.update_scene(self.data, camera=camera)
                img = r.render()
            finally:
                r.close()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue()

    def close(self):
        self._stop = True
