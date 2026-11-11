"""GLB (e.g. from Poly Pizza) -> MuJoCo-ready meshes, sized and oriented in metres, Z-up.

    from simkit.assets import glb_to_mujoco
    car = glb_to_mujoco("assets/polypizza/sedan.glb", env_dir="."  , name="sedan",
                        target={"length": 4.5}, yaw_deg=90, collision="box")
    scene.add_asset(car)                                  # <mesh>/<material> declarations
    scene.body(f'<body name="car0" pos="3 0 0">{car.geoms_xml()}</body>')

What it does:
1. loads every mesh of the glTF scene with its node transform,
2. bakes one colour per face (baseColorFactor x texture at the UV centroid x vertex colour),
3. converts glTF Y-up to Z-up (x, y, z) -> (x, -z, y) and rotates by ``yaw_deg`` about +Z,
4. scales uniformly to ``target`` ({"length"|"width"|"height"|"max": metres}) or ``scale``,
5. moves the origin to the footprint centre on the ground (``origin="base"``) or the box centre,
6. merges similar colours into <= ``max_colors`` groups and writes one OBJ per group
   (visual meshes use ``inertia="shell"`` so thin/flat parts never break compilation),
7. builds collision geometry: ``"box"`` (one fitted box), ``"hull"`` (one convex hull),
   ``"parts"`` (convex hull per connected component, good for tables/shelves/arches) or ``"none"``.

The returned ``MjAsset`` knows its real size (``.size`` = x, y, z extents in metres) and writes
``<out>/asset.json`` so build scripts can place it precisely (e.g. ``pos z = 0`` sits it on the
ground). Glb skins are shown in their bind pose (usually a T-pose).
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- colour helpers


def _srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _lab(rgb):
    lin = _srgb_to_linear(np.clip(rgb, 0, 1))
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def _face_colours(mesh) -> np.ndarray:
    """(m, 4) sRGB colour per face."""
    n = len(mesh.faces)
    vis = mesh.visual
    lin = np.full((n, 3), 0.5)
    alpha = np.ones(n)
    if getattr(vis, "kind", None) == "vertex":
        vc = np.asarray(vis.vertex_colors, np.float64)[mesh.faces].mean(axis=1) / 255.0
        return np.column_stack([vc[:, :3], vc[:, 3]])
    mat = getattr(vis, "material", None)
    if mat is not None:
        factor = getattr(mat, "baseColorFactor", None)
        image = getattr(mat, "baseColorTexture", None)
        if factor is None and image is None:  # SimpleMaterial
            factor = getattr(mat, "diffuse", None)
            image = getattr(mat, "image", None)
        f = np.ones(4) if factor is None else np.asarray(factor, np.float64).ravel()[:4]
        if f.max() > 1.0:
            f = f / 255.0
        lin = np.tile(f[:3], (n, 1))
        alpha = np.full(n, f[3] if len(f) > 3 else 1.0)
        uv = getattr(vis, "uv", None)
        if image is not None and uv is not None and len(uv) == len(mesh.vertices):
            try:
                from trimesh.visual.color import uv_to_color

                tex = uv_to_color(np.asarray(uv, np.float64)[mesh.faces].mean(axis=1), image) / 255.0
                lin = lin * _srgb_to_linear(tex[:, :3])
                alpha = alpha * tex[:, 3]
            except Exception:  # noqa: BLE001 - odd texture formats: keep the factor colour
                pass
    return np.column_stack([_linear_to_srgb(lin), alpha])


def _cluster_colours(rgba: np.ndarray, area: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge face colours into <= k groups (area-weighted, Lab distance). Returns labels, centres."""
    q = np.round(rgba[:, :3] * 24) / 24
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.ravel()
    w = np.bincount(inv, weights=area, minlength=len(uniq)) + 1e-12
    cols = np.array([np.average(rgba[inv == i, :3], axis=0, weights=area[inv == i] + 1e-12) for i in range(len(uniq))])
    groups = [[i] for i in range(len(uniq))]
    while len(groups) > k:
        cen = np.array([np.average(cols[g], axis=0, weights=w[g]) for g in groups])
        lab = _lab(cen)
        gw = np.array([w[g].sum() for g in groups])
        d = np.linalg.norm(lab[:, None] - lab[None], axis=2) + np.eye(len(groups)) * 1e9
        # merge the cheapest pair: distance weighted by the smaller group's area
        cost = d * np.minimum(gw[:, None], gw[None])
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        groups[i] += groups[j]
        groups.pop(j)
    labels = np.empty(len(uniq), int)
    for gi, g in enumerate(groups):
        labels[g] = gi
    centres = np.array([np.average(cols[g], axis=0, weights=w[g]) for g in groups])
    alphas = np.array([np.average(rgba[np.isin(inv, g), 3], weights=area[np.isin(inv, g)] + 1e-12) for g in groups])
    return labels[inv], np.column_stack([centres, alphas])


# --------------------------------------------------------------------------- mesh IO


def write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("".join(f"v {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in v))
        fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in f))


def _compact(v, f):
    used, inv = np.unique(f.ravel(), return_inverse=True)
    return v[used], inv.reshape(-1, 3)


def load_glb_faces(glb: str | Path):
    """All triangles of a GLB in its own frame: (vertices (n,3), faces (m,3), face_rgba (m,4))."""
    import trimesh

    data = Path(glb).read_bytes()
    scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene", process=False)
    vs, fs, cs, off = [], [], [], 0
    for node in scene.graph.nodes_geometry:
        tf, gname = scene.graph[node]
        g = scene.geometry.get(gname)
        if not isinstance(g, trimesh.Trimesh) or not len(g.faces):
            continue
        v = np.asarray(g.vertices, np.float64) @ tf[:3, :3].T + tf[:3, 3]
        vs.append(v)
        fs.append(np.asarray(g.faces, np.int64) + off)
        cs.append(_face_colours(g))
        off += len(v)
    if not vs:
        raise ValueError(f"{glb}: no triangle meshes")
    return np.concatenate(vs), np.concatenate(fs), np.concatenate(cs)


# --------------------------------------------------------------------------- main entry


@dataclass
class MjAsset:
    name: str
    size: tuple[float, float, float]  # x, y, z extents (m) after scaling
    visual: list[dict] = field(default_factory=list)  # {"mesh", "file", "rgba"}
    collision: list[dict] = field(default_factory=list)  # {"type": "box"|"mesh", ...}
    source: str = ""
    scale: float = 1.0

    # -- MJCF fragments -------------------------------------------------------------
    def asset_xml(self) -> str:
        out = []
        for p in self.visual:
            out.append(f'<mesh name="{p["mesh"]}" file="{p["file"]}" inertia="shell"/>')
            r, g, b, a = p["rgba"]
            peak = max(r, g, b)
            if peak > 0.86:  # near-white albedo blows out under sun + headlight: cap it, kill the shine
                r, g, b = (c * 0.86 / peak for c in (r, g, b))
            elif peak < 0.12:  # near-black reads as a featureless silhouette: lift to a dark grey
                r, g, b = ((c + 0.02) * 0.14 / (peak + 0.02) for c in (r, g, b))
            spec = 0.08 if peak > 0.8 else 0.2
            out.append(f'<material name="{p["mesh"]}_mat" rgba="{r:.3f} {g:.3f} {b:.3f} {a:.3f}" specular="{spec}" shininess="0.3"/>')
        for c in self.collision:
            if c["type"] == "mesh":
                out.append(f'<mesh name="{c["mesh"]}" file="{c["file"]}"/>')
        return "\n".join(out)

    def geoms_xml(self, collision: bool = True, visual_group: int = 2, collision_group: int = 3,
                  pos=(0, 0, 0), euler=(0, 0, 0), mass: float | None = None, friction: str | None = None,
                  rgba_override: dict | None = None, prefix: str = "") -> str:
        """Geoms to paste inside a <body>. Visual geoms do not collide and weigh nothing; the
        collision geoms carry the mass (``mass`` total, else density 1000*0.2 for props)."""
        pe = f'pos="{pos[0]:g} {pos[1]:g} {pos[2]:g}" euler="{euler[0]:g} {euler[1]:g} {euler[2]:g}"'
        lines = []
        for p in self.visual:
            mat = f'material="{p["mesh"]}_mat"'
            if rgba_override and p["mesh"] in rgba_override:
                r, g, b, a = rgba_override[p["mesh"]]
                mat = f'rgba="{r} {g} {b} {a}"'
            lines.append(f'<geom type="mesh" mesh="{p["mesh"]}" {mat} {pe} contype="0" conaffinity="0" '
                         f'group="{visual_group}" mass="0"/>')
        if collision and self.collision:
            per = None if mass is None else mass / len(self.collision)
            fr = f' friction="{friction}"' if friction else ""
            for i, c in enumerate(self.collision):
                m = f' mass="{per:g}"' if per is not None else ' density="200"'
                nm = f' name="{prefix}{self.name}_col{i}"' if prefix else ""
                if c["type"] == "box":
                    sx, sy, sz = c["size"]
                    cx, cy, cz = c["pos"]
                    lines.append(f'<geom{nm} type="box" size="{sx:.4f} {sy:.4f} {sz:.4f}" '
                                 f'pos="{cx + pos[0]:.4f} {cy + pos[1]:.4f} {cz + pos[2]:.4f}" euler="{euler[0]:g} {euler[1]:g} {euler[2]:g}" '
                                 f'group="{collision_group}" rgba="0.8 0.2 0.2 0.3"{m}{fr}/>')
                else:
                    lines.append(f'<geom{nm} type="mesh" mesh="{c["mesh"]}" {pe} group="{collision_group}" '
                                 f'rgba="0.8 0.2 0.2 0.3"{m}{fr}/>')
        return "\n".join(lines)

    def body_xml(self, name: str, pos=(0, 0, 0), euler=(0, 0, 0), free: bool = False, mass: float | None = None,
                 collision: bool = True, friction: str | None = None) -> str:
        fj = "<freejoint/>" if free else ""
        return (f'<body name="{name}" pos="{pos[0]:g} {pos[1]:g} {pos[2]:g}" euler="{euler[0]:g} {euler[1]:g} {euler[2]:g}">'
                f"{fj}\n{self.geoms_xml(collision=collision, mass=mass, friction=friction, prefix=name + '_')}\n</body>")

    def to_json(self) -> dict:
        return {"name": self.name, "size": list(self.size), "visual": self.visual, "collision": self.collision,
                "source": self.source, "scale": self.scale, "fit": getattr(self, "fit", None)}


def glb_to_mujoco(glb: str | Path, env_dir: str | Path, name: str, *, target: dict | None = None,
                  scale: float | None = None, yaw_deg: float = 0.0, up: str = "y", origin: str = "base",
                  max_colors: int = 8, collision: str = "box", max_parts: int = 12, max_stretch: float = 0.2,
                  out_subdir: str = "assets/meshes") -> MjAsset:
    """Convert ``glb`` into OBJ parts under ``env_dir/out_subdir/name`` (paths in the XML are
    relative to ``env_dir``, so keep scene.xml in ``env_dir`` and do not set compiler meshdir).

    target: {"length": m} (x extent after yaw), {"width": m} (y), {"height": m} (z),
            {"max": m} (largest extent), {"footprint": m} (max of x, y), or
            {"box": (L, W, H)}: fit a planned size - uniform scale plus at most +-max_stretch (20%)
            per axis; the result (wanted/got/stretch) is in ``asset.fit`` and asset.json, and a
            warning is printed when it still misses by > 15% (choose another model then).
    up: "y" (glTF default) or "z" if the model was authored Z-up.
    """
    env_dir = Path(env_dir).resolve()
    out = env_dir / out_subdir / name
    v, f, rgba = load_glb_faces(glb)
    if up == "y":
        v = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])
    if yaw_deg:
        a = math.radians(yaw_deg)
        rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        v = v @ rz.T
    lo, hi = v.min(axis=0), v.max(axis=0)
    ext = hi - lo
    s = 1.0
    fit_note = None
    if scale is not None:
        s = float(scale)
        v = v * s
    elif target and "box" in target:
        # fit a planned L x W x H: uniform scale (geometric mean of the per-axis ratios), then a
        # per-axis correction of at most +-max_stretch so details are not visibly distorted
        want = np.asarray(target["box"], float)
        ratios = want / np.maximum(ext, 1e-9)
        s = float(np.exp(np.mean(np.log(ratios))))
        corr = np.clip(ratios / s, 1 - max_stretch, 1 + max_stretch)
        v = v * s * corr
        got = ext * s * corr
        err = np.abs(got - want) / want
        fit_note = {"wanted": want.round(3).tolist(), "got": got.round(3).tolist(), "per_axis_stretch": corr.round(3).tolist(),
                    "max_error": round(float(err.max()), 3)}
        if err.max() > 0.15:
            import warnings

            warnings.warn(f"{name}: model proportions do not fit {want.tolist()} (got {got.round(2).tolist()}); "
                          "pick another model or build it procedurally", stacklevel=2)
    elif target:
        (k, val), = target.items()
        ref = {"length": ext[0], "width": ext[1], "height": ext[2], "max": ext.max(),
               "footprint": max(ext[0], ext[1])}[k]
        s = float(val) / max(ref, 1e-9)
        v = v * s
    else:
        v = v * s
    lo, hi = v.min(axis=0), v.max(axis=0)
    if origin == "base":
        shift = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    else:
        shift = (lo + hi) / 2
    v = v - shift
    lo, hi = v.min(axis=0), v.max(axis=0)
    size = tuple(float(x) for x in (hi - lo))

    # visual parts by colour
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    keep = area > 1e-12
    f, rgba, area = f[keep], rgba[keep], area[keep]
    labels, centres = _cluster_colours(rgba, area, max_colors)
    for old in out.glob("*.obj") if out.exists() else []:
        old.unlink()
    visual = []
    order = np.argsort([-area[labels == i].sum() for i in range(len(centres))])
    for k, ci in enumerate(order):
        sel = f[labels == ci]
        if not len(sel):
            continue
        pv, pf = _compact(v, sel)
        fn = out / f"part{k}.obj"
        write_obj(fn, pv, pf)
        visual.append({"mesh": f"{name}_v{k}", "file": str(fn.relative_to(env_dir)),
                       "rgba": [round(float(x), 4) for x in centres[ci]]})

    # collision
    col = []
    if collision == "box":
        c = (lo + hi) / 2
        col.append({"type": "box", "size": [max(float(x) / 2, 0.005) for x in (hi - lo)], "pos": [float(x) for x in c]})
    elif collision in ("hull", "parts"):
        import trimesh

        tm = trimesh.Trimesh(v, f, process=True)
        if collision == "hull":
            comps = [tm]
        else:
            try:  # connected components need scipy or networkx; fall back to one hull rather than crash
                comps = sorted(tm.split(only_watertight=False), key=lambda m: -m.bounding_box.volume) or [tm]
            except Exception:  # noqa: BLE001
                comps = [tm]
        if collision == "parts" and len(comps) > max_parts:
            rest = trimesh.util.concatenate(comps[max_parts - 1:])
            comps = comps[: max_parts - 1] + [rest]
        for i, comp in enumerate(comps):
            try:
                h = comp.convex_hull
            except Exception:  # noqa: BLE001 - degenerate component
                h = None
            ext_c = comp.bounds[1] - comp.bounds[0]
            if h is None or h.volume < 1e-7 or min(ext_c) < 0.004:
                c = comp.bounds.mean(axis=0)
                col.append({"type": "box", "size": [max(float(x) / 2, 0.003) for x in ext_c], "pos": [float(x) for x in c]})
                continue
            fn = out / f"col{i}.obj"
            write_obj(fn, np.asarray(h.vertices), np.asarray(h.faces))
            col.append({"type": "mesh", "mesh": f"{name}_c{i}", "file": str(fn.relative_to(env_dir))})
    asset = MjAsset(name=name, size=size, visual=visual, collision=col, source=str(glb), scale=s)
    asset.fit = fit_note
    out.mkdir(parents=True, exist_ok=True)
    (out / "asset.json").write_text(json.dumps(asset.to_json(), indent=1))
    return asset


def load_asset(env_dir: str | Path, name: str, out_subdir: str = "assets/meshes") -> MjAsset:
    d = json.loads((Path(env_dir) / out_subdir / name / "asset.json").read_text())
    return MjAsset(name=d["name"], size=tuple(d["size"]), visual=d["visual"], collision=d["collision"],
                   source=d.get("source", ""), scale=d.get("scale", 1.0))


def glb_to_usd(glb: str | Path, usd_path: str | Path, *, target: dict | None = None, yaw_deg: float = 0.0,
               up: str = "y", max_colors: int = 8) -> dict:
    """Same processing, written as a USD stage (Z-up, metres) with one Mesh prim per colour group
    and displayColor. For Isaac Sim scenes: reference it from the main stage and add
    UsdPhysics.CollisionAPI / RigidBodyAPI there."""
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # noqa: F401

    tmp = Path(usd_path).with_suffix("")
    a = glb_to_mujoco(glb, tmp.parent, tmp.name + "_tmp", target=target, yaw_deg=yaw_deg, up=up,
                      max_colors=max_colors, collision="none", out_subdir=".")
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/" + a.name)
    stage.SetDefaultPrim(root.GetPrim())
    for i, p in enumerate(a.visual):
        pv, pf = [], []
        for line in (tmp.parent / p["file"]).read_text().splitlines():
            if line.startswith("v "):
                pv.append(tuple(float(x) for x in line.split()[1:4]))
            elif line.startswith("f "):
                pf.append([int(x) - 1 for x in line.split()[1:4]])
        mesh = UsdGeom.Mesh.Define(stage, f"/{a.name}/part{i}")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*x) for x in pv]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(pf)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([i for t in pf for i in t]))
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*p["rgba"][:3])]))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    import shutil

    shutil.rmtree(tmp.parent / (tmp.name + "_tmp"), ignore_errors=True)
    return {"usd": str(usd_path), "size": a.size, "parts": len(a.visual)}
