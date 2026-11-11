"""Offscreen rendering helpers for smoke tests and visual checks (MuJoCo).

    from simkit.render import Views
    v = Views(model, width=960, height=540)
    imgs = v.render_all(data)                 # named cameras + auto views (overview/top/front/side/iso/robot closeups)
    v.contact_sheet(imgs, "smoke/contact_sheet.png")
    v.filmstrip(model, data, controller, seconds=4, path="smoke/filmstrip.png")

``image_stats`` flags blank/black/washed-out frames; ``visibility`` uses segmentation rendering
to count how many pixels of each top-level object are visible across the views.
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np


def _font(size=16):
    from PIL import ImageFont

    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def top_level_body(model, body_id: int) -> int:
    """The ancestor of ``body_id`` whose parent is the world (0 for world geoms)."""
    b = body_id
    while b != 0 and model.body_parentid[b] != 0:
        b = model.body_parentid[b]
    return b


def body_label(model, b: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"


def geom_world_aabbs(model, data, skip_planes=True):
    """(n, 2, 3) world AABBs of all geoms (planes skipped -> NaN)."""
    out = np.full((model.ngeom, 2, 3), np.nan)
    for g in range(model.ngeom):
        if skip_planes and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        R = data.geom_xmat[g].reshape(3, 3)
        center = data.geom_xpos[g] + R @ c
        half = np.abs(R) @ h
        out[g, 0], out[g, 1] = center - half, center + half
    return out


def scene_bounds(model, data, robust=True):
    ab = geom_world_aabbs(model, data)
    ok = ~np.isnan(ab[:, 0, 0])
    if not ok.any():
        return np.array([-1.0, -1, 0]), np.array([1.0, 1, 1])
    lo, hi = ab[ok, 0], ab[ok, 1]
    if robust and ok.sum() > 20:  # ignore a few far outliers (distant skyline, huge terrain)
        return np.percentile(lo, 2, axis=0), np.percentile(hi, 98, axis=0)
    return lo.min(0), hi.max(0)


def actuated_roots(model) -> list[int]:
    """Top-level bodies that carry actuated joints (the robots / vehicles)."""
    roots = []
    for a in range(model.nu):
        trn = model.actuator_trntype[a]
        tid = model.actuator_trnid[a, 0]
        if int(trn) in (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_JOINTINPARENT),):
            b = model.jnt_bodyid[tid]
        elif trn == mujoco.mjtTrn.mjTRN_SITE:
            b = model.site_bodyid[tid]
        elif trn == mujoco.mjtTrn.mjTRN_BODY:
            b = tid
        else:
            continue
        r = top_level_body(model, b)
        if r and r not in roots:
            roots.append(r)
    return roots


def subtree_bounds(model, data, root: int):
    ab = geom_world_aabbs(model, data)
    ids = [g for g in range(model.ngeom) if top_level_body(model, model.geom_bodyid[g]) == root and not np.isnan(ab[g, 0, 0])]
    if not ids:
        p = data.xpos[root]
        return p - 0.5, p + 0.5
    return ab[ids, 0].min(0), ab[ids, 1].max(0)


def free_camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)
    return cam


def is_mobile(model, root: int) -> bool:
    """True if the robot rooted at ``root`` can travel (free or slide joint on the root body)."""
    for j in range(model.body_jntadr[root], model.body_jntadr[root] + model.body_jntnum[root]):
        if j >= 0 and int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            return True  # free-flying or travelling on a rail/track
    return False


def pull_in_front(model, data, cam, min_frac: float = 0.25):
    """Module-level version of Views._pull_in_front for free cameras built elsewhere (showcase)."""
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    gid = np.array([-1], dtype=np.int32)
    hit = mujoco.mj_ray(model, data, np.array(cam.lookat, float), -fwd, None, 1, -1, gid)
    if min_frac * cam.distance < hit < cam.distance:
        cam.distance = max(hit * 0.88, 0.3)
    return cam


def unblock_tracking(model, data, cam, radius: float = 0.3, nominal: float | None = None):
    """Keep a tracking camera in front of scenery: if something sits between the tracked body and
    the camera, move the camera just in front of it (call every frame; ``nominal`` = wanted distance)."""
    want = nominal or cam.distance
    b = int(cam.trackbodyid)
    if b <= 0:
        return
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    back = -np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    start = data.xpos[b] + back * radius
    gid = np.array([-1], dtype=np.int32)
    hit = mujoco.mj_ray(model, data, start, back, None, 1, b, gid)
    cam.distance = max(radius + 0.9 * hit, radius * 1.2) if 0 <= hit < want - radius else want


def image_stats(img: np.ndarray) -> dict:
    g = img.astype(np.float32).mean(axis=2)
    q = (img // 16).reshape(-1, 3)
    _, counts = np.unique(q[:: max(1, len(q) // 50000)], axis=0, return_counts=True)
    dom = counts.max() / counts.sum()
    gy, gx = np.abs(np.diff(g, axis=0)), np.abs(np.diff(g, axis=1))
    edges = float(((gy[:, :-1] + gx[:-1, :]) > 25).mean())
    st = {"mean": round(float(g.mean()), 1), "std": round(float(g.std()), 1), "dominant_colour_frac": round(float(dom), 3),
          "edge_density": round(edges, 4)}
    problems = []
    if g.std() < 4 or dom > 0.97:
        problems.append("blank (almost one colour)")
    if g.mean() < 18:
        problems.append("too dark")
    if g.mean() > 245:
        problems.append("washed out")
    if edges < 0.002:
        problems.append("almost no detail (camera inside an object or looking at nothing?)")
    st["problems"] = problems
    st["ok"] = not problems
    return st


class Views:
    def __init__(self, model, width=960, height=540):
        self.model = model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.w, self.h = width, height
        self.r = mujoco.Renderer(model, height, width)
        self.opt = mujoco.MjvOption()
        # show visual groups 0-2 (simkit puts visual meshes in group 2), hide collision group 3
        for i in range(6):
            self.opt.geomgroup[i] = 1 if i <= 2 else 0

    def close(self):
        try:
            self.r.close()
        except Exception:  # noqa: BLE001
            pass

    def render(self, data, camera, segmentation=False, show_collision=False) -> np.ndarray:
        opt = self.opt
        if show_collision:
            opt = mujoco.MjvOption()
            for i in range(6):
                opt.geomgroup[i] = 1
        if segmentation:
            self.r.enable_segmentation_rendering()
        try:
            self.r.update_scene(data, camera=camera, scene_option=opt)
            return self.r.render().copy()
        finally:
            if segmentation:
                self.r.disable_segmentation_rendering()

    def auto_views(self, data, max_robots=3) -> dict:
        """Free cameras framing the scene: overview (iso), top, front, side + closeups of robots."""
        lo, hi = scene_bounds(self.model, data)
        c = (lo + hi) / 2
        span = float(np.linalg.norm(hi - lo))
        fovy = 45.0
        d = max(span * 0.62 / math.tan(math.radians(fovy / 2)) * 0.55, 0.6)
        views = {
            "auto_overview": free_camera(c, d * 1.05, 135, -28),
            "auto_top": free_camera(c, d * 1.1, 90, -89),
            "auto_front": free_camera(c, d, 180, -12),
            "auto_side": free_camera(c, d, 90, -12),
            "auto_iso_back": free_camera(c, d, -45, -30),
        }
        for i, root in enumerate(actuated_roots(self.model)[:max_robots]):
            blo, bhi = subtree_bounds(self.model, data, root)
            bc = (blo + bhi) / 2
            bs = max(float(np.linalg.norm(bhi - blo)), 0.3)
            views[f"auto_robot_{body_label(self.model, root)}"] = free_camera(bc, bs * 1.9, 150, -22)
        for name, cam in views.items():  # never leave an auto camera behind a wall / inside a building
            if name.startswith("auto_") and name != "auto_top":
                self._pull_in_front(data, cam)
                self._swing_until_detailed(data, cam)
        return views

    def _swing_until_detailed(self, data, cam):
        """If the view renders blank/featureless (e.g. the back of a room wall), try other azimuths."""
        if image_stats(self.render(data, cam))["ok"]:
            return
        az0 = cam.azimuth
        for daz in (60, -60, 120, -120, 180, 30, -30):
            cam.azimuth = az0 + daz
            if image_stats(self.render(data, cam))["ok"]:
                return
        cam.azimuth = az0

    def _pull_in_front(self, data, cam):
        az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
        fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        start = np.array(cam.lookat, float)
        gid = np.array([-1], dtype=np.int32)
        hit = mujoco.mj_ray(self.model, data, start, -fwd, None, 1, -1, gid)
        if 0.25 * cam.distance < hit < cam.distance:  # an occluder well away from the target
            cam.distance = max(hit * 0.88, 0.3)

    def _route_camera(self, model, data, root, controller_factory, seconds):
        try:
            d = mujoco.MjData(model)
            mujoco.mj_copyData(d, model, data)
            ctrl = controller_factory()
            pts = [d.xpos[root].copy()]
            every = max(1, int(0.1 / model.opt.timestep))
            for k in range(int(seconds / model.opt.timestep)):
                if ctrl is not None:
                    ctrl(model, d)
                mujoco.mj_step(model, d)
                if k % every == 0:
                    if not np.isfinite(d.qpos).all():
                        return None
                    pts.append(d.xpos[root].copy())
            pts = np.array(pts)
            lo, hi = pts.min(0), pts.max(0)
            blo, bhi = subtree_bounds(model, data, root)
            size = float(np.linalg.norm(bhi - blo))
            span = float(np.linalg.norm(hi - lo))
            if span < 2 * size:  # robot barely moves: the overview is fine
                return None
            cam = free_camera((lo + hi) / 2, max(span * 0.75, size * 4), 120, -58)  # steep: few foreground occluders
            self._pull_in_front(data, cam)
            return cam
        except Exception:  # noqa: BLE001 - framing is best effort
            return None

    def overview_camera(self, data):
        """The scene's own fixed overview camera if it has one (indoor scenes: inside the walls),
        else the automatic overview."""
        world_cams = [c for c in range(self.model.ncam) if self.model.cam_bodyid[c] == 0]
        named = {c: (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, c) or "").lower() for c in world_cams}
        for key in ("overview", "wide", "main", "top", "birdseye"):
            for c, n in named.items():
                if key in n:
                    return c
        if world_cams:
            return world_cams[0]
        return self.auto_views(data)["auto_overview"]

    def named_cameras(self) -> dict:
        return {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam{i}": i for i in range(self.model.ncam)}

    def render_all(self, data, include_auto=True, include_named=True) -> dict[str, np.ndarray]:
        out = {}
        if include_named:
            for name, cid in self.named_cameras().items():
                out[f"cam_{name}"] = self.render(data, cid)
        if include_auto:
            for name, cam in self.auto_views(data).items():
                out[name] = self.render(data, cam)
        return out

    def visibility(self, data, cameras: dict, reduce: str = "sum") -> dict[str, int]:
        """Pixels per top-level object across ``cameras`` (segmentation render).
        ``reduce="sum"``: total over views; ``"max"``: best single view (how big it ever appears)."""
        counts: dict[str, int] = {}
        for cam in cameras.values():
            view: dict[str, int] = {}
            seg = self.render(data, cam, segmentation=True)
            ids, types = seg[..., 0].ravel(), seg[..., 1].ravel()
            geoms = ids[(types == mujoco.mjtObj.mjOBJ_GEOM) & (ids >= 0)]
            if not len(geoms):
                continue
            gid, n = np.unique(geoms, return_counts=True)
            for g, k in zip(gid, n):
                if g >= self.model.ngeom:
                    continue
                b = top_level_body(self.model, self.model.geom_bodyid[g])
                if b == 0:
                    nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "world_geom"
                    key = f"world:{nm}"
                else:
                    key = body_label(self.model, b)
                view[key] = view.get(key, 0) + int(k)
            for key, k in view.items():
                counts[key] = counts.get(key, 0) + k if reduce == "sum" else max(counts.get(key, 0), k)
        return counts

    # -- composition ------------------------------------------------------------------
    @staticmethod
    def contact_sheet(images: dict[str, np.ndarray], path, cols=3, tile_w=640, title: str | None = None):
        from PIL import Image, ImageDraw

        names = list(images)
        if not names:
            return None
        first = images[names[0]]
        tile_h = int(tile_w * first.shape[0] / first.shape[1])
        rows = (len(names) + cols - 1) // cols
        top = 40 if title else 0
        sheet = Image.new("RGB", (cols * tile_w, top + rows * (tile_h + 26)), (24, 24, 28))
        draw = ImageDraw.Draw(sheet)
        if title:
            draw.text((12, 10), title, fill=(240, 240, 240), font=_font(20))
        for i, n in enumerate(names):
            im = Image.fromarray(images[n]).resize((tile_w, tile_h))
            x, y = (i % cols) * tile_w, top + (i // cols) * (tile_h + 26)
            sheet.paste(im, (x, y + 26))
            draw.text((x + 8, y + 4), n, fill=(230, 230, 230), font=_font(16))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path)
        return path

    def filmstrip(self, model, data, controller=None, seconds=4.0, frames=8, camera=None, path=None, video_path=None, fps=30,
                  controller_factory=None):
        """Roll the sim forward with ``controller(model, data)`` (sets data.ctrl) and capture frames.

        Two rows per time step when a robot exists: a tracking closeup and a FIXED wide view, so
        motion is visible against the world. Returns the list of (t, {view: image}).
        Also writes an MP4 (tracking view) if ``video_path`` and imageio-ffmpeg are available.
        """
        views = {}
        roots = actuated_roots(model)
        if camera is not None:
            views["camera"] = camera
        elif roots and not is_mobile(model, roots[0]):
            # fixed-base robot (arm on a table): a tracking camera would sit inside its own links;
            # frame the workspace from outside instead
            blo, bhi = subtree_bounds(model, data, roots[0])
            span = max(float(np.linalg.norm(bhi - blo)), 0.5)
            ws = free_camera((blo + bhi) / 2, span * 1.6, 135, -28)
            self._pull_in_front(data, ws)
            views["workspace"] = ws
            views["scene_overview"] = self.overview_camera(data)
        elif roots:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = roots[0]
            blo, bhi = subtree_bounds(model, data, roots[0])
            cam.distance = max(float(np.linalg.norm(bhi - blo)) * 2.5, 1.0)
            cam.azimuth, cam.elevation = 135, -20
            views["tracking"] = cam
            # second row: a fixed camera framing the ROUTE the robot actually drives (cheap physics-only
            # pre-run with a fresh controller), so travel stays in frame and the robot is not a speck
            route = self._route_camera(model, data, roots[0], controller_factory, seconds) if controller_factory else None
            views["route" if route is not None else "scene_overview"] = route if route is not None else self.overview_camera(data)
        else:
            views["overview"] = self.auto_views(data)["auto_overview"]
        n_steps = int(round(seconds / model.opt.timestep))
        shot_steps = set(np.linspace(0, n_steps, frames).astype(int).tolist())
        vid_every = max(1, int(round(1.0 / (fps * model.opt.timestep))))
        first = next(iter(views.values()))
        shots, video = [], []
        nominal = {n: getattr(c, "distance", 0.0) for n, c in views.items()}
        for k in range(n_steps + 1):
            for n, c in views.items():
                if isinstance(c, mujoco.MjvCamera) and c.type == mujoco.mjtCamera.mjCAMERA_TRACKING and (k in shot_steps or (video_path and k % vid_every == 0)):
                    unblock_tracking(model, data, c, nominal=nominal[n])
            if k in shot_steps:
                shots.append((data.time, {n: self.render(data, c) for n, c in views.items()}))
            if video_path and k % vid_every == 0:
                video.append(self.render(data, first))
            if k == n_steps:
                break
            if controller is not None:
                controller(model, data)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all():
                break
        if path:
            tiles = {}
            for vn in views:
                for t, ims in shots:
                    tiles[f"{vn} t={t:.2f}s"] = ims[vn]
            self.contact_sheet(tiles, path, cols=len(shots), tile_w=360)
        if video_path and video:
            try:
                import imageio.v2 as iio

                iio.mimwrite(video_path, video, fps=fps, quality=7, macro_block_size=1)
            except Exception as e:  # noqa: BLE001 - ffmpeg missing: frames are enough
                (Path(video_path).with_suffix(".txt")).write_text(f"video not written: {e}")
        return shots
