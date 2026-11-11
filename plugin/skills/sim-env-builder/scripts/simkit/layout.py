"""Layout sanity: things a physics rollout cannot see because static geoms never collide.

* floating   static objects with nothing underneath (gap > 3 cm)
* sunken     objects whose bottom is well below the local ground surface
* overlap    static objects interpenetrating (a tree inside a house, two shelves merged)
* off_ground objects placed beyond the edge of a finite ground plane
* scale      sizes compared with ``expected_sizes`` {object: [min_h, max_h]} if given

"Objects" are top-level bodies (children of the world) plus individual geoms attached directly to
the world body; the ground (largest plane/hfield) and flat ground-like geoms (roads, sidewalks,
lane paint, water surfaces) are skipped.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np

from .physics_exam import chk, ground_geom
from .render import body_label, geom_world_aabbs, top_level_body


def objects(model, data):
    """{label: {"geoms": [...], "lo", "hi", "static": bool}}"""
    ab = geom_world_aabbs(model, data)
    gg = ground_geom(model)
    objs: dict[str, dict] = {}
    for g in range(model.ngeom):
        if g == gg or np.isnan(ab[g, 0, 0]) or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            continue
        if model.geom_rgba[g, 3] == 0 and model.geom_contype[g] == 0:
            continue
        b = top_level_body(model, model.geom_bodyid[g])
        if b == 0:
            key = "geom:" + (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"#{g}")
        else:
            key = body_label(model, b)
        o = objs.setdefault(key, {"geoms": [], "lo": ab[g, 0].copy(), "hi": ab[g, 1].copy(), "body": b})
        o["geoms"].append(g)
        o["lo"] = np.minimum(o["lo"], ab[g, 0])
        o["hi"] = np.maximum(o["hi"], ab[g, 1])
    for o in objs.values():
        b = o["body"]
        o["static"] = b == 0 or not any(model.body_dofnum[x] > 0 for x in range(model.nbody) if top_level_body(model, x) == b)
        ext = o["hi"] - o["lo"]
        o["flat"] = ext[2] < 0.06 and ext[0] * ext[1] > 1.0  # roads, rugs, paint, water sheets
    return objs


def _ray_down(model, data, x, y, z):
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, np.array([x, y, z], float), np.array([0, 0, -1.0]), None, 1, -1, gid)
    return dist, int(gid[0])


def ground_height(model, data, x, y, top):
    """Surface height at (x, y): first hit of a ray cast straight down from ``top``."""
    d, g = _ray_down(model, data, x, y, top)
    return (top - d) if d >= 0 else None


def _support_distance(model, data, o, ab, gg, cand_mask):
    """Smallest signed distance from the object's geoms to any other geom near its bottom.

    Uses mj_geomDistance (exact for primitives and convex meshes); heightfields are handled
    with rays because geom distance does not support them.
    """
    lo, hi = o["lo"], o["hi"]
    own = set(o["geoms"])
    best = np.inf
    near = np.nonzero(cand_mask & (ab[:, 1, 0] >= lo[0] - 0.05) & (ab[:, 0, 0] <= hi[0] + 0.05)
                      & (ab[:, 1, 1] >= lo[1] - 0.05) & (ab[:, 0, 1] <= hi[1] + 0.05)
                      & (ab[:, 1, 2] >= lo[2] - 0.1) & (ab[:, 0, 2] <= lo[2] + 0.35 * (hi[2] - lo[2]) + 0.05))[0]
    cands = [int(g) for g in near if g not in own]
    if gg >= 0 and model.geom_type[gg] == mujoco.mjtGeom.mjGEOM_PLANE:
        cands.append(gg)
    low_geoms = sorted(own, key=lambda g: ab[g, 0, 2])[:12]
    for g1 in low_geoms:
        if int(model.geom_type[g1]) in (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD),):
            continue
        for g2 in cands:
            if model.geom_type[g2] == mujoco.mjtGeom.mjGEOM_HFIELD:
                continue
            try:
                dd = mujoco.mj_geomDistance(model, data, g1, g2, 0.5, None)
            except Exception:  # noqa: BLE001 - unsupported pair
                continue
            best = min(best, dd)
    if gg >= 0 and model.geom_type[gg] == mujoco.mjtGeom.mjGEOM_HFIELD:
        cx, cy = (lo[:2] + hi[:2]) / 2
        h = ground_height(model, data, cx, cy, hi[2] + 50)
        if h is not None:
            best = min(best, lo[2] - h)
    return best


def _min_distance(model, data, geoms_a, geoms_b, cap: int = 60):
    """Most negative signed distance between two geom sets (None if no supported pair)."""
    best = None
    skip = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
    ga = [g for g in geoms_a if int(model.geom_type[g]) not in skip][:cap]
    gb = [g for g in geoms_b if int(model.geom_type[g]) not in skip][:cap]
    for g1 in ga:
        for g2 in gb:
            try:
                dd = mujoco.mj_geomDistance(model, data, g1, g2, 0.1, None)
            except Exception:  # noqa: BLE001
                continue
            best = dd if best is None else min(best, dd)
    return best


def _ground_distance(model, data, o, gg):
    if gg < 0 or model.geom_type[gg] != mujoco.mjtGeom.mjGEOM_PLANE:
        return None
    best = np.inf
    for g1 in o["geoms"]:
        if int(model.geom_type[g1]) in (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD),):
            continue
        try:
            best = min(best, mujoco.mj_geomDistance(model, data, g1, gg, 0.5, None))
        except Exception:  # noqa: BLE001
            continue
    return best if np.isfinite(best) else None


SUSPENDED = re.compile(r"(^|[_:])(mounted|hanging|ceiling|suspended|overhead)[_]?", re.I)  # intentionally off the floor


def check_layout(model, data, expected_sizes: dict | None = None, allow_floating=()):
    out = []
    mujoco.mj_forward(model, data)
    objs = objects(model, data)
    gg = ground_geom(model)
    ab = geom_world_aabbs(model, data)
    cand_mask = ~np.isnan(ab[:, 0, 0])
    floating, sunken, off = [], [], []
    for name, o in objs.items():
        if not o["static"] or o["flat"] or any(a in name for a in allow_floating) or SUSPENDED.search(name):
            continue
        lo, hi = o["lo"], o["hi"]
        cx, cy = (lo[:2] + hi[:2]) / 2
        gap = _support_distance(model, data, o, ab, gg, cand_mask)
        if gap > 0.03:
            floating.append(f"{name} (gap {gap:.2f} m)" if np.isfinite(gap) else f"{name} (nothing below)")
        gd = _ground_distance(model, data, o, gg)
        if gd is not None and gd < -0.05 and (hi[2] - lo[2]) > 0.1:
            sunken.append(f"{name} ({-gd:.2f} m below the ground plane)")
        if gg >= 0 and model.geom_type[gg] == mujoco.mjtGeom.mjGEOM_PLANE:
            sx, sy = model.geom_size[gg, :2]
            gx, gy = data.geom_xpos[gg][:2]
            if sx > 0 and (abs(cx - gx) > sx or abs(cy - gy) > sy):
                off.append(name)
    out.append(chk("floating_objects", "layout", "warn" if floating else "pass", floating[:15] or 0, "static props rest on something",
                   details="static objects with a gap underneath (intended for hanging lamps/signs? then ignore)"))
    out.append(chk("sunken_objects", "layout", "warn" if sunken else "pass", sunken[:15] or 0, "bottom at ground level",
                   details="object bottoms more than 5 cm below the surrounding ground"))
    if off:
        out.append(chk("off_ground", "layout", "warn", off[:15], details="objects outside the ground plane"))
    # static-static overlaps (AABB intersection relative to the smaller box)
    names = [n for n, o in objs.items() if o["static"] and not o["flat"]]
    over = []
    if 1 < len(names) < 1500:
        lo = np.array([objs[n]["lo"] for n in names])
        hi = np.array([objs[n]["hi"] for n in names])
        vol = np.prod(np.maximum(hi - lo, 1e-3), axis=1)
        for i in range(len(names)):
            ilo = np.maximum(lo[i], lo[i + 1:])
            ihi = np.minimum(hi[i], hi[i + 1:])
            inter = np.prod(np.clip(ihi - ilo, 0, None), axis=1)
            frac = inter / np.minimum(vol[i], vol[i + 1:])
            for k in np.nonzero(frac > 0.0)[0]:
                over.append((float(frac[k]), names[i], names[i + 1 + k]))
    # boxes overlap often harmlessly (a table under a tree canopy): confirm with exact geom distances
    real = []
    veg = re.compile(r"tree|bush|shrub|plant|hedge|grass|foliage|flower|palm|pine|canopy|leaf", re.I)
    for f, a, b in sorted(over, reverse=True)[:400]:
        if veg.search(a) and veg.search(b):
            continue  # overlapping canopies are natural
        depth = _min_distance(model, data, objs[a]["geoms"], objs[b]["geoms"])
        if depth is not None and depth < -0.02:
            real.append((-depth, a, b))
    real.sort(reverse=True)
    out.append(chk("static_overlaps", "layout", "warn" if real else "pass", [f"{a} x {b} ({d * 100:.0f} cm deep)" for d, a, b in real[:12]] or 0,
                   "no interpenetrating props", details="static props whose geometry interpenetrates by > 2 cm (exact geom distance; may be intended for nested items)"))
    if expected_sizes:
        bad = []
        for name, (mn, mx) in expected_sizes.items():
            o = objs.get(name)
            if o is None:
                bad.append(f"{name}: not found")
                continue
            h = float(o["hi"][2] - o["lo"][2])
            if not (mn <= h <= mx):
                bad.append(f"{name}: height {h:.2f} m not in [{mn}, {mx}]")
        out.append(chk("real_world_scale", "layout", "warn" if bad else "pass", bad or 0, "heights within expected ranges"))
    return out
