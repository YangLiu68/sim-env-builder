"""Procedural fallback assets: anything Poly Pizza does not have, built from code.

Every recipe returns a ``Prop`` sized in metres with its origin at the footprint centre on the
ground (z = 0 is the bottom), so ``body_xml(pos=(x, y, ground_z))`` sits it on the surface.
Primitive-based props collide with their own geoms (good, cheap contacts). Mesh-based props
(``lathe``, ``extrude``, ``rock``) write OBJ files under ``env_dir/assets/procedural``.

    from simkit import procedural as P
    tbl = P.table(1.4, 0.8, 0.75)
    sc.add_asset(tbl); sc.body(tbl.body_xml("table", pos=(0, 0, 0)))
    gate = P.gate(width=1.6, height=1.6, post_height=1.0)
    rock = P.rock(env_dir, "rock3", radius=0.6, seed=3)
    terrain = P.terrain(name="hills", size=(40, 40), max_height=2.5, kind="hills", seed=1)

Available recipes (all keyword-tunable):
  nature:    tree(kind="round"|"pine"|"palm"), bush, rock (mesh), grass_patch
  building:  building, house, wall, fence, door, window_panel, stairs, ramp, pillar
  street:    traffic_cone, traffic_light, street_lamp, stop_sign, bench, bollard, barrier, car_body
  indoor:    table, chair, desk, shelf, cabinet, sofa, bed, tv, lamp, plant_pot, counter
  logistics: crate, pallet, barrel, conveyor, rack
  tabletop:  mug, bottle, bowl, plate, can, fruit(kind="apple"|"orange"|"banana"|"lemon"), block
  sport/air: gate (drone gate), ring_gate, ball, goal, landing_pad
  water:     buoy, dock
  terrain:   terrain (heightfield: hills|rough|craters|dunes|steps), road(length, lanes)
  deformable: cloth_xml (flexcomp grid cloth; tested in MuJoCo 4.0, real-time at 11x11)
  meshes:    lathe(profile), extrude(polygon, height), rock
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .assets import write_obj

Color = tuple[float, float, float, float]

PALETTE = {
    "wood": (0.55, 0.38, 0.22, 1), "dark_wood": (0.35, 0.22, 0.12, 1), "metal": (0.62, 0.64, 0.68, 1),
    "dark_metal": (0.25, 0.26, 0.28, 1), "white": (0.92, 0.92, 0.90, 1), "black": (0.08, 0.08, 0.09, 1),
    "red": (0.80, 0.15, 0.12, 1), "orange": (0.95, 0.45, 0.08, 1), "yellow": (0.95, 0.80, 0.15, 1),
    "green": (0.25, 0.55, 0.22, 1), "leaf": (0.22, 0.48, 0.20, 1), "pine": (0.13, 0.36, 0.20, 1),
    "bark": (0.40, 0.27, 0.17, 1), "blue": (0.15, 0.35, 0.75, 1), "concrete": (0.62, 0.62, 0.60, 1),
    "brick": (0.62, 0.30, 0.22, 1), "glass": (0.55, 0.70, 0.80, 0.55), "fabric": (0.35, 0.40, 0.55, 1),
    "cardboard": (0.72, 0.56, 0.36, 1), "rubber": (0.10, 0.10, 0.10, 1), "stone": (0.50, 0.49, 0.47, 1),
}


def _c(color) -> str:
    if isinstance(color, str):
        color = PALETTE[color]
    r, g, b, *a = color
    return f"{r:.3f} {g:.3f} {b:.3f} {a[0] if a else 1:.3f}"


def geom(type_: str, size, pos=(0, 0, 0), color="concrete", euler=None, fromto=None, collide=True,
         mass=None, density=None, friction=None, name=None, extra="") -> str:
    """One MJCF geom string. ``collide=False`` makes it purely visual (and weightless)."""
    s = " ".join(f"{x:.4f}" for x in np.atleast_1d(size))
    loc = f'fromto="{" ".join(f"{x:.4f}" for x in fromto)}"' if fromto is not None else f'pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"'
    eu = f' euler="{euler[0]:g} {euler[1]:g} {euler[2]:g}"' if euler is not None else ""
    col = "" if collide else ' contype="0" conaffinity="0" group="2" mass="0"'
    m = f' mass="{mass:g}"' if (mass is not None and collide) else (f' density="{density:g}"' if (density is not None and collide) else "")
    fr = f' friction="{friction}"' if friction else ""
    nm = f' name="{name}"' if name else ""
    return f'<geom{nm} type="{type_}" size="{s}" {loc}{eu} rgba="{_c(color)}"{col}{m}{fr} {extra}/>'.replace("  ", " ")


@dataclass
class Prop:
    name: str
    geoms: list[str]
    size: tuple[float, float, float]
    assets: list[str] = field(default_factory=list)
    notes: str = ""

    def asset_xml(self) -> str:
        return "\n".join(self.assets)

    def geoms_xml(self, prefix: str = "") -> str:
        if not prefix:
            return "\n".join(self.geoms)
        out = []
        for g in self.geoms:  # give named geoms a unique prefix when the prop is instanced
            out.append(g.replace(' name="', f' name="{prefix}'))
        return "\n".join(out)

    def body_xml(self, name: str, pos=(0, 0, 0), euler=(0, 0, 0), free: bool = False, mass: float | None = None) -> str:
        fj = "<freejoint/>" if free else ""
        inertial = ""
        if mass is not None:
            sx, sy, sz = self.size
            inertial = (f'<inertial pos="0 0 {sz / 2:.4f}" mass="{mass:g}" diaginertia="{mass * (sy**2 + sz**2) / 12:.5f} '
                        f'{mass * (sx**2 + sz**2) / 12:.5f} {mass * (sx**2 + sy**2) / 12:.5f}"/>')
        return (f'<body name="{name}" pos="{pos[0]:g} {pos[1]:g} {pos[2]:g}" euler="{euler[0]:g} {euler[1]:g} {euler[2]:g}">'
                f"{fj}{inertial}\n{self.geoms_xml(prefix=name + '_')}\n</body>")


# =========================================================================== mesh generators

def lathe_mesh(profile: list[tuple[float, float]], segments: int = 32):
    """Revolve (radius, z) points around +Z. Endpoints with radius 0 close the shape."""
    prof = [(max(r, 0.0), z) for r, z in profile]
    verts, rings = [], []
    for r, z in prof:
        if r < 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
        else:
            ring = []
            for k in range(segments):
                a = 2 * math.pi * k / segments
                ring.append(len(verts))
                verts.append((r * math.cos(a), r * math.sin(a), z))
            rings.append(ring)
    faces = []
    for r0, r1 in zip(rings[:-1], rings[1:]):
        if len(r0) == 1 and len(r1) == 1:
            continue
        for k in range(segments):
            if len(r0) == 1:
                faces.append((r0[0], r1[(k + 1) % segments], r1[k]))
            elif len(r1) == 1:
                faces.append((r0[k], r0[(k + 1) % segments], r1[0]))
            else:
                a, b = r0[k], r0[(k + 1) % segments]
                c, d = r1[k], r1[(k + 1) % segments]
                faces += [(a, b, d), (a, d, c)]
    return np.array(verts), np.array(faces)


def _ear_clip(poly: np.ndarray) -> list[tuple[int, int, int]]:
    idx = list(range(len(poly)))
    area = sum(poly[i, 0] * poly[(i + 1) % len(poly), 1] - poly[(i + 1) % len(poly), 0] * poly[i, 1] for i in idx)
    if area < 0:
        idx.reverse()
    tris, guard = [], 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for i in range(len(idx)):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
            pa, pb, pc = poly[a], poly[b], poly[c]
            if (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]) <= 1e-12:
                continue
            inside = False
            for j in idx:
                if j in (a, b, c):
                    continue
                p = poly[j]
                d1 = (pb[0] - pa[0]) * (p[1] - pa[1]) - (pb[1] - pa[1]) * (p[0] - pa[0])
                d2 = (pc[0] - pb[0]) * (p[1] - pb[1]) - (pc[1] - pb[1]) * (p[0] - pb[0])
                d3 = (pa[0] - pc[0]) * (p[1] - pc[1]) - (pa[1] - pc[1]) * (p[0] - pc[0])
                if d1 >= 0 and d2 >= 0 and d3 >= 0:
                    inside = True
                    break
            if not inside:
                tris.append((a, b, c))
                idx.pop(i)
                break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def extrude_mesh(polygon: list[tuple[float, float]], height: float):
    """Prism from a simple (possibly concave) polygon in the XY plane, from z=0 to z=height."""
    p = np.asarray(polygon, float)
    n = len(p)
    tris = _ear_clip(p)
    v = np.vstack([np.column_stack([p, np.zeros(n)]), np.column_stack([p, np.full(n, height)])])
    f = [(c, b, a) for a, b, c in tris] + [(a + n, b + n, c + n) for a, b, c in tris]
    ccw = sum(p[i, 0] * p[(i + 1) % n, 1] - p[(i + 1) % n, 0] * p[i, 1] for i in range(n)) > 0
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, j + n), (i, j + n, i + n)] if ccw else [(j, i, i + n), (j, i + n, j + n)]
    return v, np.array(f)


def rock_mesh(radius: float = 0.5, seed: int = 0, roughness: float = 0.25, flatten: float = 0.6, subdiv: int = 2):
    """Lumpy convex-ish rock (icosphere + low-frequency radial noise), base flattened at z=0."""
    rng = np.random.default_rng(seed)
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
                  [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdiv):
        cache, nf, vl = {}, [], list(v)

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = (vl[a] + vl[b]) / 2
                vl.append(m / np.linalg.norm(m))
                cache[key] = len(vl) - 1
            return cache[key]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        v, f = np.array(vl), np.array(nf)
    dirs = rng.normal(size=(6, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    amp = rng.uniform(0.3, 1.0, 6) * roughness
    r = 1 + (np.maximum(v @ dirs.T, 0) ** 2 * amp).sum(1) - amp.sum() * 0.3
    v = v * r[:, None] * radius
    v[:, 2] *= flatten
    v[:, 2] = np.maximum(v[:, 2], v[:, 2].min() * 0.55)  # flat bottom so it rests
    v[:, 2] -= v[:, 2].min()
    return v, f


def mesh_prop(env_dir, name: str, v: np.ndarray, f: np.ndarray, color="stone", collide: bool = True,
              density: float = 2500.0, subdir: str = "assets/procedural") -> Prop:
    """Write an OBJ and wrap it as a Prop (collision = convex hull of the mesh)."""
    env_dir = Path(env_dir).resolve()
    v = np.asarray(v, float).copy()
    v[:, :2] -= (v[:, :2].min(0) + v[:, :2].max(0)) / 2
    v[:, 2] -= v[:, 2].min()
    fn = env_dir / subdir / f"{name}.obj"
    write_obj(fn, v, np.asarray(f, int))
    rel = fn.relative_to(env_dir)
    ext = v.max(0) - v.min(0)
    col = f' density="{density:g}"' if collide else ' contype="0" conaffinity="0" group="2" mass="0"'
    return Prop(name, [f'<geom type="mesh" mesh="{name}_mesh" rgba="{_c(color)}"{col}/>'], tuple(float(x) for x in ext),
                assets=[f'<mesh name="{name}_mesh" file="{rel}" inertia="{"convex" if collide else "shell"}"/>'])


def rock(env_dir, name: str = "rock", radius: float = 0.5, seed: int = 0, color="stone", roughness=0.25) -> Prop:
    v, f = rock_mesh(radius, seed, roughness)
    return mesh_prop(env_dir, name, v, f, color)


def lathe(env_dir, name: str, profile, color="white", segments=32, collide=True, density=800.0) -> Prop:
    v, f = lathe_mesh(profile, segments)
    return mesh_prop(env_dir, name, v, f, color, collide, density)


def extrude(env_dir, name: str, polygon, height: float, color="concrete", collide=True, density=1000.0) -> Prop:
    v, f = extrude_mesh(polygon, height)
    return mesh_prop(env_dir, name, v, f, color, collide, density)


# =========================================================================== nature

def tree(height: float = 5.0, kind: str = "round", seed: int = 0, trunk_color="bark", leaf_color=None) -> Prop:
    rng = np.random.default_rng(seed)
    h = height
    tr = max(0.06, 0.035 * h)
    if kind == "pine":
        leaf = leaf_color or "pine"
        g = [geom("cylinder", (tr, h * 0.2), (0, 0, h * 0.2), trunk_color)]
        for i, (zf, rf) in enumerate([(0.30, 0.30), (0.50, 0.24), (0.68, 0.17), (0.84, 0.10)]):
            r = rf * h * rng.uniform(0.92, 1.08)
            g.append(geom("ellipsoid", (r, r, h * 0.16), (0, 0, h * zf + h * 0.06), leaf, collide=i == 0))
        return Prop("tree_pine", g, (0.6 * h, 0.6 * h, h))
    if kind == "palm":
        g = [geom("capsule", (tr * 0.8,), fromto=(0, 0, 0, h * 0.08, 0, h * 0.9), color=trunk_color)]
        for k in range(7):
            a = 2 * math.pi * k / 7 + rng.uniform(-0.2, 0.2)
            g.append(geom("ellipsoid", (h * 0.22, h * 0.05, h * 0.015), (h * 0.08 + math.cos(a) * h * 0.18, math.sin(a) * h * 0.18, h * 0.86),
                          leaf_color or "leaf", euler=(0, 15, math.degrees(a)), collide=False))
        return Prop("tree_palm", g, (0.5 * h, 0.5 * h, h))
    leaf = leaf_color or "leaf"
    g = [geom("cylinder", (tr, h * 0.3), (0, 0, h * 0.3), trunk_color)]
    cr = 0.3 * h
    g.append(geom("ellipsoid", (cr, cr * rng.uniform(0.9, 1.1), cr * 0.85), (0, 0, h - cr * 0.85), leaf))
    for _ in range(3):
        a = rng.uniform(0, 2 * math.pi)
        g.append(geom("sphere", (cr * 0.55,), (math.cos(a) * cr * 0.6, math.sin(a) * cr * 0.6, h - cr * 1.1), leaf, collide=False))
    return Prop("tree_round", g, (2 * cr, 2 * cr, h))


def bush(radius: float = 0.6, color="leaf", seed: int = 0) -> Prop:
    rng = np.random.default_rng(seed)
    g = [geom("ellipsoid", (radius, radius, radius * 0.7), (0, 0, radius * 0.6), color)]
    for _ in range(3):
        a = rng.uniform(0, 2 * math.pi)
        g.append(geom("sphere", (radius * 0.55,), (math.cos(a) * radius * 0.6, math.sin(a) * radius * 0.6, radius * 0.45), color, collide=False))
    return Prop("bush", g, (2 * radius, 2 * radius, radius * 1.3))


def grass_patch(size=(2.0, 2.0), color="green") -> Prop:
    return Prop("grass_patch", [geom("box", (size[0] / 2, size[1] / 2, 0.005), (0, 0, 0.005), color, collide=False)], (size[0], size[1], 0.01))


# =========================================================================== buildings

def building(width=12.0, depth=12.0, height=20.0, color="concrete", window_color="glass", floors: int | None = None) -> Prop:
    w, d, h = width / 2, depth / 2, height
    g = [geom("box", (w, d, h / 2), (0, 0, h / 2), color)]
    floors = floors or max(1, int(h / 3.2))
    fh = h / floors
    for i in range(floors):
        z = fh * (i + 0.55)
        for sgn in (-1, 1):
            g.append(geom("box", (w * 0.86, 0.02, fh * 0.22), (0, sgn * (d + 0.01), z), window_color, collide=False))
            g.append(geom("box", (0.02, d * 0.86, fh * 0.22), (sgn * (w + 0.01), 0, z), window_color, collide=False))
    g.append(geom("box", (w * 1.01, d * 1.01, 0.15), (0, 0, h + 0.15), "dark_metal", collide=False))
    return Prop("building", g, (width, depth, h + 0.3))


def house(width=8.0, depth=10.0, wall_height=3.0, roof_height=2.0, wall_color=(0.88, 0.84, 0.74, 1),
          roof_color="brick") -> Prop:
    w, d = width / 2, depth / 2
    g = [geom("box", (w, d, wall_height / 2), (0, 0, wall_height / 2), wall_color)]
    slope = math.degrees(math.atan2(roof_height, w))
    L = math.hypot(w, roof_height)
    for s in (-1, 1):
        g.append(geom("box", (L / 2 + 0.2, d + 0.3, 0.08), (s * w / 2, 0, wall_height + roof_height / 2), roof_color, euler=(0, s * slope, 0)))
    g.append(geom("box", (0.6, 0.05, 1.05), (0, -d - 0.03, 1.05), "dark_wood", collide=False))
    for x in (-w * 0.55, w * 0.55):
        g.append(geom("box", (0.6, 0.04, 0.5), (x, -d - 0.03, 1.7), "glass", collide=False))
    return Prop("house", g, (width, depth, wall_height + roof_height))


def wall(length=4.0, height=2.5, thickness=0.15, color="concrete") -> Prop:
    return Prop("wall", [geom("box", (length / 2, thickness / 2, height / 2), (0, 0, height / 2), color)], (length, thickness, height))


def fence(length=4.0, height=1.1, post_spacing=1.0, color="wood") -> Prop:
    n = max(2, int(length / post_spacing) + 1)
    g = [geom("box", (0.04, 0.04, height / 2), (-length / 2 + i * length / (n - 1), 0, height / 2), color) for i in range(n)]
    for z in (height * 0.35, height * 0.85):
        g.append(geom("box", (length / 2, 0.015, 0.05), (0, 0, z), color))
    return Prop("fence", g, (length, 0.08, height))


def door(width=0.9, height=2.05, color="dark_wood") -> Prop:
    return Prop("door", [geom("box", (width / 2, 0.02, height / 2), (0, 0, height / 2), color),
                         geom("sphere", (0.03,), (width * 0.38, -0.05, height * 0.48), "metal", collide=False)], (width, 0.04, height))


def window_panel(width=1.2, height=1.2, color="glass") -> Prop:
    return Prop("window", [geom("box", (width / 2, 0.01, height / 2), (0, 0, height / 2), color)], (width, 0.02, height))


def stairs(steps=6, rise=0.17, run=0.28, width=1.2, color="concrete") -> Prop:
    g = [geom("box", (run / 2, width / 2, rise * (i + 1) / 2), (i * run + run / 2, 0, rise * (i + 1) / 2), color) for i in range(steps)]
    return Prop("stairs", g, (steps * run, width, steps * rise), notes="ascends along +x from x=0")


def ramp(length=3.0, height=0.5, width=1.5, color="concrete") -> Prop:
    ang = math.degrees(math.atan2(height, length))
    L = math.hypot(length, height)
    return Prop("ramp", [geom("box", (L / 2, width / 2, 0.05), (0, 0, height / 2), color, euler=(0, -ang, 0))], (length, width, height),
                notes="rises along +x")


def pillar(radius=0.3, height=4.0, color="concrete") -> Prop:
    return Prop("pillar", [geom("cylinder", (radius, height / 2), (0, 0, height / 2), color)], (2 * radius, 2 * radius, height))


# =========================================================================== street

def traffic_cone(height=0.7, color="orange") -> Prop:
    g = [geom("box", (0.18, 0.18, 0.015), (0, 0, 0.015), color),
         geom("cylinder", (0.11, height * 0.45), (0, 0, height * 0.47), color),
         geom("cylinder", (0.07, height * 0.06), (0, 0, height * 0.86), "white", collide=False)]
    return Prop("traffic_cone", g, (0.36, 0.36, height))


def traffic_light(height=5.0, arm=0.0) -> Prop:
    g = [geom("cylinder", (0.08, height / 2), (0, 0, height / 2), "dark_metal"),
         geom("box", (0.17, 0.15, 0.5), (arm, 0.0, height + 0.3), "black")]
    for i, col in enumerate([(0.9, 0.1, 0.1, 1), (0.95, 0.75, 0.1, 1), (0.1, 0.85, 0.3, 1)]):
        g.append(geom("sphere", (0.09,), (arm, -0.16, height + 0.62 - 0.32 * (i + 0.5)), col, collide=False, name=["lamp_red", "lamp_amber", "lamp_green"][i]))
    if arm:
        g.append(geom("box", (abs(arm) / 2, 0.05, 0.05), (arm / 2, 0, height - 0.05), "dark_metal"))
    return Prop("traffic_light", g, (0.4 + abs(arm), 0.4, height + 0.8), notes="named geoms lamp_red/amber/green: change rgba to switch")


def street_lamp(height=6.0) -> Prop:
    g = [geom("cylinder", (0.07, height / 2), (0, 0, height / 2), "dark_metal"),
         geom("box", (0.6, 0.04, 0.04), (0.6, 0, height), "dark_metal"),
         geom("box", (0.25, 0.12, 0.05), (1.1, 0, height - 0.05), (1, 0.95, 0.8, 1), collide=False)]
    return Prop("street_lamp", g, (1.4, 0.3, height))


def stop_sign(height=2.2) -> Prop:
    g = [geom("cylinder", (0.03, height / 2), (0, 0, height / 2), "metal"),
         geom("cylinder", (0.38, 0.01), (0, -0.04, height), "red", euler=(90, 0, 0)),
         geom("cylinder", (0.33, 0.012), (0, -0.045, height), "white", euler=(90, 0, 0), collide=False),
         geom("cylinder", (0.30, 0.013), (0, -0.05, height), "red", euler=(90, 0, 0), collide=False)]
    return Prop("stop_sign", g, (0.76, 0.1, height + 0.38))


def bench(length=1.6, color="wood") -> Prop:
    g = [geom("box", (length / 2, 0.22, 0.03), (0, 0, 0.45), color),
         geom("box", (length / 2, 0.03, 0.2), (0, 0.22, 0.7), color, euler=(10, 0, 0))]
    for x in (-length / 2 + 0.1, length / 2 - 0.1):
        g.append(geom("box", (0.03, 0.22, 0.21), (x, 0, 0.21), "dark_metal"))
    return Prop("bench", g, (length, 0.5, 0.9))


def bollard(height=1.0, color="dark_metal") -> Prop:
    return Prop("bollard", [geom("cylinder", (0.1, height / 2), (0, 0, height / 2), color),
                            geom("cylinder", (0.102, 0.04), (0, 0, height * 0.8), "yellow", collide=False)], (0.2, 0.2, height))


def barrier(length=2.0, height=1.0) -> Prop:
    g = [geom("box", (length / 2, 0.25, 0.05), (0, 0, 0.05), "concrete"),
         geom("box", (length / 2, 0.12, height / 2 - 0.05), (0, 0, height / 2 + 0.05), "concrete")]
    for i in range(int(length / 0.5)):
        g.append(geom("box", (0.12, 0.125, 0.08), (-length / 2 + 0.25 + i * 0.5, 0, height * 0.7), "red" if i % 2 else "white", collide=False))
    return Prop("barrier", g, (length, 0.5, height))


def car_body(length=4.5, color="blue", width: float | None = None) -> Prop:
    """Static/visual car shape (no wheel joints). For a drivable car build an articulated body."""
    L, W = length, width or length * 0.42
    H = L * 0.32
    wr = L * 0.075
    g = [geom("box", (L / 2, W / 2, H * 0.22), (0, 0, wr + H * 0.2), color),
         geom("box", (L * 0.27, W / 2 * 0.9, H * 0.2), (-L * 0.05, 0, wr + H * 0.6), color),
         geom("box", (L * 0.24, W / 2 * 0.92, H * 0.15), (-L * 0.05, 0, wr + H * 0.62), "glass", collide=False)]
    for x in (L * 0.32, -L * 0.32):
        for y in (W / 2 - 0.12, -W / 2 + 0.12):
            g.append(geom("cylinder", (wr, 0.11), (x, y, wr), "rubber", euler=(90, 0, 0)))
    for y in (W / 2 - 0.2, -W / 2 + 0.2):
        g.append(geom("box", (0.02, 0.12, 0.05), (L / 2, y, wr + H * 0.25), (1, 1, 0.85, 1), collide=False))
    return Prop("car_body", g, (L, W, wr + H * 0.8), notes="front faces +x")


def road(length=50.0, lanes=2, lane_width=3.5, z=0.0, asphalt=(0.2, 0.2, 0.22, 1), shoulder=0.5) -> Prop:
    """Flat road strip along +x with lane markings (centre double yellow, dashed white between)."""
    w = lanes * lane_width + 2 * shoulder
    g = [geom("box", (length / 2, w / 2, 0.01), (0, 0, 0.01), asphalt, friction="1.0 0.005 0.0001")]
    half = lanes // 2
    for i in range(1, lanes):
        y = -lanes * lane_width / 2 + i * lane_width
        if lanes % 2 == 0 and i == half:
            for dy in (-0.1, 0.1):
                g.append(geom("box", (length / 2, 0.06, 0.002), (0, y + dy, 0.021), "yellow", collide=False))
        else:
            n = int(length / 6)
            for k in range(n):
                g.append(geom("box", (1.5, 0.06, 0.002), (-length / 2 + 3 + k * 6, y, 0.021), "white", collide=False))
    for s in (-1, 1):
        g.append(geom("box", (length / 2, 0.07, 0.002), (0, s * (lanes * lane_width / 2), 0.021), "white", collide=False))
    return Prop("road", g, (length, w, 0.022), notes="surface at z=0.02; markings are visual only")


# =========================================================================== indoor

def table(width=1.2, depth=0.8, height=0.75, color="wood", leg_color=None) -> Prop:
    t = 0.04
    g = [geom("box", (width / 2, depth / 2, t / 2), (0, 0, height - t / 2), color)]
    for x in (width / 2 - 0.06, -width / 2 + 0.06):
        for y in (depth / 2 - 0.06, -depth / 2 + 0.06):
            g.append(geom("box", (0.03, 0.03, (height - t) / 2), (x, y, (height - t) / 2), leg_color or color))
    return Prop("table", g, (width, depth, height), notes=f"top surface at z={height}")


def chair(seat_height=0.45, color="wood") -> Prop:
    s = 0.22
    g = [geom("box", (s, s, 0.025), (0, 0, seat_height), color),
         geom("box", (0.02, s, 0.22), (-s + 0.02, 0, seat_height + 0.24), color)]
    for x in (s - 0.03, -s + 0.03):
        for y in (s - 0.03, -s + 0.03):
            g.append(geom("box", (0.02, 0.02, seat_height / 2), (x, y, seat_height / 2), color))
    return Prop("chair", g, (2 * s, 2 * s, seat_height + 0.47), notes="faces +x (backrest at -x)")


def desk(width=1.4, depth=0.7, height=0.74) -> Prop:
    p = table(width, depth, height, "white", "dark_metal")
    p.geoms.append(geom("box", (0.2, depth / 2 - 0.05, 0.25), (width / 2 - 0.25, 0, height - 0.3), "white"))
    p.name = "desk"
    return p


def shelf(width=1.0, depth=0.4, height=2.0, levels=5, color="metal") -> Prop:
    g = []
    for i in range(levels):
        z = 0.08 + i * (height - 0.1) / (levels - 1)
        g.append(geom("box", (width / 2, depth / 2, 0.015), (0, 0, z), color))
    for x in (width / 2 - 0.02, -width / 2 + 0.02):
        for y in (depth / 2 - 0.02, -depth / 2 + 0.02):
            g.append(geom("box", (0.02, 0.02, height / 2), (x, y, height / 2), "dark_metal"))
    return Prop("shelf", g, (width, depth, height), notes=f"{levels} shelves from z=0.08 to {height - 0.02:.2f}")


def rack(width=2.7, depth=1.1, height=4.0, levels=3) -> Prop:
    g = []
    for i in range(levels):
        z = 0.15 + i * height / levels
        for y in (depth / 2 - 0.05, -depth / 2 + 0.05):
            g.append(geom("box", (width / 2, 0.05, 0.06), (0, y, z), "orange"))
        g.append(geom("box", (width / 2, depth / 2, 0.01), (0, 0, z + 0.07), "metal"))
    for x in (width / 2 - 0.05, -width / 2 + 0.05):
        for y in (depth / 2 - 0.05, -depth / 2 + 0.05):
            g.append(geom("box", (0.04, 0.04, height / 2), (x, y, height / 2), "blue"))
    return Prop("pallet_rack", g, (width, depth, height))


def cabinet(width=0.8, depth=0.5, height=0.9, color="white") -> Prop:
    return Prop("cabinet", [geom("box", (width / 2, depth / 2, height / 2), (0, 0, height / 2), color),
                            geom("box", (0.01, 0.1, 0.01), (width * 0.0, -depth / 2 - 0.02, height * 0.8), "metal", collide=False, euler=(0, 0, 90))],
                (width, depth, height))


def counter(width=2.0, depth=0.6, height=0.92) -> Prop:
    return Prop("counter", [geom("box", (width / 2, depth / 2, (height - 0.04) / 2), (0, 0, (height - 0.04) / 2), "white"),
                            geom("box", (width / 2 + 0.02, depth / 2 + 0.02, 0.02), (0, 0, height - 0.02), "stone")], (width, depth, height))


def sofa(width=2.0, depth=0.9, height=0.85, color="fabric") -> Prop:
    g = [geom("box", (width / 2, depth / 2, 0.2), (0, 0, 0.2), color),
         geom("box", (width / 2, 0.12, 0.25), (0, depth / 2 - 0.12, 0.6), color)]
    for s in (-1, 1):
        g.append(geom("box", (0.1, depth / 2, 0.15), (s * (width / 2 - 0.1), 0, 0.55), color))
    g.append(geom("box", (width / 2 - 0.2, depth / 2 - 0.15, 0.06), (0, -0.08, 0.46), (0.42, 0.47, 0.62, 1), collide=False))
    return Prop("sofa", g, (width, depth, height), notes="faces -y (backrest at +y)")


def bed(width=1.6, length=2.1, color="white") -> Prop:
    g = [geom("box", (length / 2, width / 2, 0.15), (0, 0, 0.25), "dark_wood"),
         geom("box", (length / 2 - 0.02, width / 2 - 0.02, 0.1), (0, 0, 0.5), color),
         geom("box", (0.05, width / 2, 0.45), (length / 2, 0, 0.45), "dark_wood"),
         geom("box", (0.18, width / 2 - 0.15, 0.06), (length / 2 - 0.3, 0, 0.64), (0.95, 0.95, 0.97, 1), collide=False)]
    return Prop("bed", g, (length, width, 0.9))


def tv(width=1.2, color="black") -> Prop:
    h = width * 9 / 16
    g = [geom("box", (width / 2, 0.03, h / 2), (0, 0, 0.12 + h / 2 + 0.05), color),
         geom("box", (0.2, 0.12, 0.01), (0, 0, 0.01), "dark_metal"),
         geom("box", (0.03, 0.02, 0.09), (0, 0, 0.1), "dark_metal")]
    return Prop("tv", g, (width, 0.24, h + 0.17))


def lamp(height=1.6) -> Prop:
    g = [geom("cylinder", (0.15, 0.015), (0, 0, 0.015), "dark_metal"),
         geom("cylinder", (0.015, height / 2), (0, 0, height / 2), "dark_metal"),
         geom("cylinder", (0.18, 0.12), (0, 0, height), (0.98, 0.94, 0.82, 1), collide=False)]
    return Prop("floor_lamp", g, (0.36, 0.36, height + 0.12))


def plant_pot(height=0.9) -> Prop:
    g = [geom("cylinder", (0.18, 0.17), (0, 0, 0.17), "brick"),
         geom("ellipsoid", (0.3, 0.3, height * 0.35), (0, 0, 0.34 + height * 0.3), "leaf")]
    return Prop("plant_pot", g, (0.6, 0.6, height))


# =========================================================================== logistics

def crate(size=(0.6, 0.4, 0.4), color="cardboard") -> Prop:
    sx, sy, sz = size
    g = [geom("box", (sx / 2, sy / 2, sz / 2), (0, 0, sz / 2), color),
         geom("box", (sx / 2 + 0.001, 0.02, sz / 2 + 0.001), (0, 0, sz / 2), (0.62, 0.48, 0.30, 1), collide=False)]
    return Prop("crate", g, size)


def pallet(length=1.2, width=1.0, color="wood") -> Prop:
    g = [geom("box", (length / 2, width / 2, 0.011), (0, 0, 0.144 - 0.011), color)]
    for y in (-width / 2 + 0.05, 0, width / 2 - 0.05):
        g.append(geom("box", (length / 2, 0.05, 0.061), (0, y, 0.061), color))
    return Prop("pallet", g, (length, width, 0.144))


def barrel(radius=0.29, height=0.88, color="blue") -> Prop:
    g = [geom("cylinder", (radius, height / 2), (0, 0, height / 2), color)]
    for z in (height * 0.3, height * 0.7):
        g.append(geom("cylinder", (radius + 0.005, 0.01), (0, 0, z), "dark_metal", collide=False))
    return Prop("barrel", g, (2 * radius, 2 * radius, height))


def conveyor(length=4.0, width=0.6, height=0.8) -> Prop:
    g = [geom("box", (length / 2, width / 2, 0.05), (0, 0, height - 0.05), "rubber", name="belt", friction="1.2 0.005 0.0001"),
         geom("box", (length / 2, 0.03, 0.08), (0, width / 2 + 0.03, height - 0.02), "dark_metal"),
         geom("box", (length / 2, 0.03, 0.08), (0, -width / 2 - 0.03, height - 0.02), "dark_metal")]
    for x in np.linspace(-length / 2 + 0.2, length / 2 - 0.2, 4):
        for y in (width / 2 - 0.03, -width / 2 + 0.03):
            g.append(geom("box", (0.03, 0.03, (height - 0.1) / 2), (x, y, (height - 0.1) / 2), "metal"))
    return Prop("conveyor", g, (length, width + 0.12, height),
                notes="belt top at z=height. To move objects: make the belt a body on a slide joint or set velocity of a kinematic mocap; simplest is to apply a velocity to items on the belt each step.")


# =========================================================================== tabletop

def mug(env_dir, name="mug", radius=0.04, height=0.1, color="white") -> Prop:
    t = 0.006
    prof = [(0, 0), (radius, 0), (radius, height), (radius - t, height), (radius - t, t), (0, t)]
    p = lathe(env_dir, name, prof, color, density=900)
    p.geoms.append(geom("capsule", (0.008,), fromto=(radius, 0, height * 0.75, radius + 0.03, 0, height * 0.5), color=color, mass=0.02))
    p.geoms.append(geom("capsule", (0.008,), fromto=(radius + 0.03, 0, height * 0.5, radius, 0, height * 0.25), color=color, mass=0.02))
    return p


def bottle(env_dir, name="bottle", radius=0.035, height=0.25, color=(0.2, 0.55, 0.3, 0.85)) -> Prop:
    h = height
    prof = [(0, 0), (radius, 0), (radius, h * 0.6), (radius * 0.45, h * 0.8), (radius * 0.35, h * 0.97), (radius * 0.38, h), (0, h)]
    return lathe(env_dir, name, prof, color, density=400)


def bowl(env_dir, name="bowl", radius=0.08, height=0.06, color="white") -> Prop:
    t = 0.005
    prof = [(0, 0), (radius * 0.5, 0), (radius, height), (radius - t, height), (radius * 0.5 - t, t), (0, t)]
    return lathe(env_dir, name, prof, color, density=900)


def plate(radius=0.12, color="white") -> Prop:
    return Prop("plate", [geom("cylinder", (radius, 0.008), (0, 0, 0.008), color, mass=0.3)], (2 * radius, 2 * radius, 0.016))


def can(radius=0.033, height=0.122, color="red") -> Prop:
    return Prop("can", [geom("cylinder", (radius, height / 2), (0, 0, height / 2), color, mass=0.35)], (2 * radius, 2 * radius, height))


def fruit(kind="apple") -> Prop:
    if kind == "banana":
        g = [geom("capsule", (0.018,), fromto=(-0.08, 0, 0.03, 0, 0, 0.018), color="yellow", mass=0.06),
             geom("capsule", (0.018,), fromto=(0, 0, 0.018, 0.08, 0, 0.03), color="yellow", mass=0.06)]
        return Prop("banana", g, (0.2, 0.036, 0.05))
    spec = {"apple": (0.038, (0.75, 0.12, 0.1, 1), 0.18), "orange": (0.037, (0.97, 0.55, 0.1, 1), 0.2),
            "lemon": (0.03, (0.97, 0.88, 0.25, 1), 0.12)}[kind]
    r, col, m = spec
    g = [geom("sphere", (r,), (0, 0, r), col, mass=m)]
    if kind == "apple":
        g.append(geom("capsule", (0.003,), fromto=(0, 0, 2 * r - 0.004, 0.003, 0, 2 * r + 0.012), color="bark", collide=False))
    return Prop(kind, g, (2 * r, 2 * r, 2 * r))


def block(size=0.05, color="red", mass=0.1) -> Prop:
    return Prop("block", [geom("box", (size / 2,) * 3, (0, 0, size / 2), color, mass=mass)], (size, size, size))


# =========================================================================== sport / aerial / water

def gate(width=1.6, height=1.6, post_height=1.0, thickness=0.08, color="orange") -> Prop:
    """Square drone-racing gate on two posts; opening centre at z = post_height + height/2, facing +-x."""
    t = thickness
    zc = post_height + height / 2
    g = [geom("box", (t / 2, width / 2 + t, t / 2), (0, 0, post_height), color),
         geom("box", (t / 2, width / 2 + t, t / 2), (0, 0, post_height + height), color),
         geom("box", (t / 2, t / 2, height / 2), (0, -width / 2 - t / 2, zc), color),
         geom("box", (t / 2, t / 2, height / 2), (0, width / 2 + t / 2, zc), color)]
    for s in (-1, 1):
        g.append(geom("cylinder", (0.04, post_height / 2), (0, s * (width / 2 + t / 2), post_height / 2), "dark_metal"))
    return Prop("gate", g, (t, width + 2 * t, post_height + height + t / 2), notes=f"fly through along x; opening centre z={zc:.2f}")


def ring_gate(radius=0.8, tube=0.05, post_height=1.0, segments=16, color="red") -> Prop:
    zc = post_height + radius
    g = []
    for k in range(segments):
        a0, a1 = 2 * math.pi * k / segments, 2 * math.pi * (k + 1) / segments
        g.append(geom("capsule", (tube,), fromto=(0, radius * math.cos(a0), zc + radius * math.sin(a0), 0, radius * math.cos(a1), zc + radius * math.sin(a1)), color=color))
    g.append(geom("cylinder", (0.04, (post_height - tube) / 2), (0, 0, (post_height - tube) / 2), "dark_metal"))
    return Prop("ring_gate", g, (2 * tube, 2 * radius + 2 * tube, post_height + 2 * radius + tube), notes=f"ring centre z={zc:.2f}, axis along x")


def ball(radius=0.11, color="white", mass=0.43) -> Prop:
    return Prop("ball", [geom("sphere", (radius,), (0, 0, radius), color, mass=mass, extra='condim="6"')], (2 * radius,) * 3,
                notes="use condim 6 for rolling friction")


def goal(width=3.0, height=2.0, depth=1.0, color="white") -> Prop:
    g = [geom("cylinder", (0.05, height / 2), (0, s * width / 2, height / 2), color) for s in (-1, 1)]
    g.append(geom("cylinder", (0.05, width / 2), (0, 0, height), color, euler=(90, 0, 0)))
    g.append(geom("box", (0.01, width / 2, height / 2), (depth, 0, height / 2), (0.9, 0.9, 0.9, 0.35), collide=False))
    return Prop("goal", g, (depth, width + 0.1, height + 0.05))


def landing_pad(radius=1.0) -> Prop:
    g = [geom("cylinder", (radius, 0.02), (0, 0, 0.02), "dark_metal"),
         geom("cylinder", (radius * 0.8, 0.002), (0, 0, 0.041), "yellow", collide=False),
         geom("cylinder", (radius * 0.72, 0.002), (0, 0, 0.0415), "dark_metal", collide=False),
         geom("box", (radius * 0.08, radius * 0.4, 0.002), (-radius * 0.25, 0, 0.042), "white", collide=False),
         geom("box", (radius * 0.08, radius * 0.4, 0.002), (radius * 0.25, 0, 0.042), "white", collide=False),
         geom("box", (radius * 0.25, radius * 0.08, 0.002), (0, 0, 0.0425), "white", collide=False)]
    return Prop("landing_pad", g, (2 * radius, 2 * radius, 0.045))


def buoy(radius=0.4, color="red") -> Prop:
    g = [geom("cylinder", (radius, radius * 0.6), (0, 0, radius * 0.6), color),
         geom("cone" if False else "cylinder", (radius * 0.3, radius * 0.6), (0, 0, radius * 1.8), color),
         geom("sphere", (radius * 0.15,), (0, 0, radius * 2.5), "yellow", collide=False)]
    return Prop("buoy", g, (2 * radius, 2 * radius, radius * 2.65))


def dock(length=10.0, width=2.5, height=1.0) -> Prop:
    g = [geom("box", (length / 2, width / 2, 0.08), (0, 0, height), "wood")]
    for x in np.linspace(-length / 2 + 0.3, length / 2 - 0.3, max(2, int(length / 2.5) + 1)):
        for y in (-width / 2 + 0.15, width / 2 - 0.15):
            g.append(geom("cylinder", (0.12, (height + 1.0) / 2), (x, y, (height - 1.0) / 2), "dark_wood"))
    return Prop("dock", g, (length, width, height + 0.08), notes="piles extend 1 m below z=0 (into water)")


# =========================================================================== terrain

def terrain(name="terrain", size=(40.0, 40.0), max_height=2.0, kind="hills", seed=0, res=None, base=0.5,
            color=(0.42, 0.5, 0.3, 1), flat_center: float = 0.0, edge_blend: float = 0.12) -> Prop:
    """Heightfield terrain centred at the origin. kind: hills | rough | craters | dunes | steps.
    Keep a ground plane (``sc.ground``) larger than the tile: ``edge_blend`` (fraction of the tile)
    tapers the border to height 0 so the tile blends into it instead of ending in a cliff.

    ``flat_center`` (m): radius around the origin that is flattened to the mean level, handy for
    spawn areas. Heights are in [0, max_height]; the hfield bottom is ``base`` m below z=0.
    """
    rng = np.random.default_rng(seed)
    n = int(res) if res else int(np.clip(max(size) * 2.5, 128, 400))  # ~40 cm cells; finer for big terrain
    y, x = np.mgrid[0:n, 0:n] / (n - 1)
    h = np.zeros((n, n))
    if kind in ("hills", "dunes", "rough"):
        octaves = {"hills": [(2, 1.0), (4, 0.45), (8, 0.15)], "dunes": [(3, 1.0), (6, 0.2)],
                   "rough": [(4, 0.6), (10, 0.5), (22, 0.35)]}[kind]
        for freq, amp in octaves:
            for _ in range(3):
                kx, ky = rng.normal(size=2) * freq
                ph = rng.uniform(0, 2 * np.pi)
                h += amp * np.sin(2 * np.pi * (kx * x + ky * y) + ph)
        if kind == "dunes":
            h = np.abs(h)
    elif kind == "craters":
        h += 0.15 * rng.normal(size=(n, n)).cumsum(0).cumsum(1) * 0
        for _ in range(int(rng.integers(6, 14))):
            cx, cy, r = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(0.04, 0.15)
            d = np.hypot(x - cx, y - cy) / r
            h += np.where(d < 1, -(1 - d**2), 0) + 0.35 * np.exp(-((d - 1) ** 2) / 0.05)
        h += 0.08 * np.sin(13 * x + 2) * np.sin(11 * y + 1)
    elif kind == "steps":
        h = np.floor(x * 10) / 10 + 0.02 * rng.normal(size=(n, n))
    h -= h.min()
    h /= max(h.max(), 1e-9)
    if edge_blend > 0:  # taper heights to 0 at the tile border so it meets the ground plane, no cliff into the void
        ex = np.clip(np.minimum(x, 1 - x) / edge_blend, 0, 1)
        ey = np.clip(np.minimum(y, 1 - y) / edge_blend, 0, 1)
        w = np.minimum(ex, ey)
        h = h * (w * w * (3 - 2 * w))
    if flat_center > 0:
        r = np.hypot((x - 0.5) * size[0], (y - 0.5) * size[1])
        w = np.clip((r - flat_center) / max(flat_center, 1e-6), 0, 1)
        h = h * w + h.mean() * (1 - w)
    elev = " ".join(f"{v:.4f}" for v in h.ravel())
    assets = [f'<hfield name="{name}" nrow="{n}" ncol="{n}" size="{size[0] / 2} {size[1] / 2} {max_height} {base}" elevation="{elev}"/>']
    g = [f'<geom name="{name}" type="hfield" hfield="{name}" rgba="{_c(color)}" friction="1 0.005 0.0001"/>']
    return Prop(name, g, (size[0], size[1], max_height), assets=assets,
                notes="height at (x, y) = simkit.procedural.hfield_height(model, name, x, y); spawn objects above it")


def hfield_height(model, name: str, x: float, y: float) -> float:
    """Terrain surface z at world (x, y) for a heightfield geom created by ``terrain`` (geom at origin)."""
    import mujoco

    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, name)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    sx, sy, sz, _ = model.hfield_size[hid]
    nr, nc = model.hfield_nrow[hid], model.hfield_ncol[hid]
    data = model.hfield_data[model.hfield_adr[hid]: model.hfield_adr[hid] + nr * nc].reshape(nr, nc)
    gx, gy, gz = model.geom_pos[gid]
    u = np.clip((x - gx + sx) / (2 * sx) * (nc - 1), 0, nc - 1)
    v = np.clip((y - gy + sy) / (2 * sy) * (nr - 1), 0, nr - 1)
    i0, j0 = int(v), int(u)
    i1, j1 = min(i0 + 1, nr - 1), min(j0 + 1, nc - 1)
    fu, fv = u - j0, v - i0
    h = (data[i0, j0] * (1 - fu) * (1 - fv) + data[i0, j1] * fu * (1 - fv) + data[i1, j0] * (1 - fu) * fv + data[i1, j1] * fu * fv)
    return float(gz + h * sz)


# =========================================================================== deformables

def cloth_xml(name="cloth", size=(0.4, 0.4), n=11, pos=(0, 0, 1.0), mass=0.12, rgba=(0.2, 0.4, 0.8, 1),
              radius=0.004, damping=0.002, solref="0.003 1") -> str:
    """Flexcomp grid cloth (put the string in the worldbody). ``n`` vertices per side: 9-15 keeps RL
    real-time (11x11 runs ~4x real time on a laptop); finer grids look nicer but get slow.
    Spawn it ``radius`` above its support (pos z = support top + radius). Vertex bodies are named
    ``{name}_<i>``; use ``cloth_vertex_bodies(model, name)`` to read them (e.g. a corner to grasp).
    Edges are equality-constrained (inextensible-ish); the shell-bending plugin is not in the pip build."""
    sx, sy = size
    sp = (sx / (n - 1), sy / (n - 1))
    return (f'<flexcomp name="{name}" type="grid" count="{n} {n} 1" spacing="{sp[0]:.5f} {sp[1]:.5f} 0.01" '
            f'pos="{pos[0]:g} {pos[1]:g} {pos[2]:g}" radius="{radius}" rgba="{_c(rgba)}" dim="2" mass="{mass}">'
            f'<edge equality="true" damping="{damping}"/>'
            f'<contact selfcollide="none" solref="{solref}" condim="3"/></flexcomp>')


def cloth_vertex_bodies(model, name="cloth"):
    """Body ids of a flexcomp's vertices (row-major, n x n), via the flex's vertex list."""
    import mujoco

    f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
    adr, num = model.flex_vertadr[f], model.flex_vertnum[f]
    return [int(b) for b in model.flex_vertbodyid[adr: adr + num]]
