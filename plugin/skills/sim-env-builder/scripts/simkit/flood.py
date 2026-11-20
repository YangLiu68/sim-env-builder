"""Flooded streets, yards and fords: a water depth field, wading physics, manhole fountains, storm look.

    from simkit import flood as F
    spec = F.FloodSpec(
        regions=[(-20, -6, 20, 6), (-6, -20, 6, 20)],          # rectangles (x0, y0, x1, y1) where water stands
        depth=0.12,                                              # base depth, metres
        bands=[{"axis": "y", "lo": -6, "hi": -2.5, "depth": 0.38}],   # deeper lane / gutter side
        pools=[{"center": [3, -4], "radius": 3, "depth": 0.1}],       # extra depth in a dip
        flow=(0.0, -0.6),                                        # current, m/s
        geysers=[F.Geyser("mh1", (2.0, 1.5), height=0.8)],       # manhole fountains
    )
    # build_env.py
    F.storm_look(sc)                     # grey sky, fog, soft light (instead of sc.sky(...))
    F.add_to_scene(sc, spec, env_dir=HERE, wakes=["ego", "npc1"])   # water, foam, bow waves, spec in scene.xml
    # world.py
    from simkit import flood as F
    def make_world(model):
        flood = F.FloodWorld(model)      # reads the spec from the model; vehicles auto-detected
        def world(model, data):
            flood(model, data)           # sets xfrc_applied of flooded vehicles: ADD your own forces after it
            ...                          # traffic lights, NPC drivers
        return world

Physics (approximate; every term has a physical form and a stated parameter):

  buoyancy   rho*g*fill*(submerged hull volume), per quarter of the footprint, so a car on a depth
             gradient pitches/rolls. fill=0.65 (cars are not watertight), underside 0.22 m: a 2 t car
             loses most of its tyre load at ~0.6 m of water, as real cars do (0.5-0.6 m).
  drag       -0.5*rho*Cd*A*|v_rel|*v_rel, v_rel = vehicle velocity - water velocity, A = immersed
             frontal area: wheels/underbody up to the ground clearance (blockage 0.35), full
             projected width above it. Cd = 1.0.
  grip       tyre friction mu = mu_dry*(1 - 0.35*min(depth/0.3, 1)) at each wheel (water film); the
             buoyancy also unloads the tyres, which is what makes cars slide in deep water.
  flow       uniform current inside the flood + radial outflow around fountains (acts via drag).
  fountains  surcharged manholes: jet speed v = sqrt(2*g*h) for fountain height h; the momentum
             flux rho*A_open*v^2 pushes up on whatever covers the manhole, at the manhole.

Depth is measured from the road surface ``bed_z`` (flat road); the water top is bed_z + depth.
Depth differences of 0.1-0.4 m over a street come from the crown, gutters and dips of a real road
and from the flowing water's surface slope (~1 %): invisible at street scale, so the road stays flat
for contact and only the water surface follows the depth field.

Check helpers: ``wheel_load`` (tyre normal force from contacts), ``coast_test`` (speed lost
coasting through water of a given depth), ``fountain_uplift`` (tyre load with the fountain on vs
off), ``find_spot`` (a point with a given depth), ``FloodWorld.info`` (per-vehicle depth/forces).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

G = 9.81
SPEC_TEXT = "simkit_flood"
WATER_RGBA = (0.40, 0.37, 0.32, 0.94)     # muddy grey-brown storm water
FOAM_RGBA = (0.88, 0.87, 0.83, 1.0)   # foam is slightly grey, never paper white


@dataclass
class Geyser:
    """A surcharged manhole: water erupts as a pulsing foam dome (the cover lifted or blown off)."""
    name: str
    pos: tuple
    height: float = 0.7        # fountain height above the water surface at mean strength (m)
    radius: float = 1.0        # foam mound radius (m)
    period: float = 2.4        # pulsing period (s)
    pulse: float = 0.35        # +- fraction of the height
    phase: float = 0.0         # rad
    open_area: float = 0.1     # effective opening of the lifted cover (m^2); a 0.6 m manhole is 0.28 m^2
    outflow: float = 0.8       # radial surface current around the fountain (m/s)
    start: float = 0.0         # s: the manhole erupts at this time (0 = from the start; > 0 = a surprise
                               # eruption for reactive-driving tests: no baked foam, nothing visible before)
    ramp: float = 0.8          # s: from cover lifting to full jet

    def strength(self, t: float) -> float:
        """0..(1+pulse): multiplies the height. A slow beat plus a faster flutter, never negative;
        0 before ``start``, ramping up over ``ramp`` seconds."""
        if t < self.start:
            return 0.0
        s = 1.0 + self.pulse * (0.75 * math.sin(2 * math.pi * t / self.period + self.phase)
                                + 0.25 * math.sin(2 * math.pi * t / (0.37 * self.period) + 2.1 * self.phase + 1.0))
        if self.ramp > 0 and t < self.start + self.ramp:
            s *= (t - self.start) / self.ramp
        return max(0.0, s)

    def active(self, t: float) -> bool:
        """Erupting now (what a driver can see)."""
        return self.strength(t) > 0.05

    def jet_force(self, t: float, rho: float = 1000.0) -> float:
        """Upward momentum flux (N) of the jet: rho * A * v^2 with v = sqrt(2 g h)."""
        h = self.height * self.strength(t)
        return rho * self.open_area * 2 * G * h


@dataclass
class FloodSpec:
    regions: list = field(default_factory=list)   # [(x0, y0, x1, y1), ...] rectangles, union = flooded area
    depth: float = 0.15                            # base depth (m)
    bands: list = field(default_factory=list)      # [{"axis": "x"|"y", "lo": a, "hi": b, "depth": d}] lanes/gutters
    pools: list = field(default_factory=list)      # [{"center": [x, y], "radius": r, "depth": +d}] dips (gaussian)
    edge: float = 1.5                              # shoreline taper width (m)
    blend: float = 0.8                             # band edge blend width (m)
    flow: tuple = (0.0, 0.0)                       # current (m/s)
    bed_z: float = 0.0                             # road surface height under the water (m)
    geysers: list = field(default_factory=list)    # [Geyser, ...]
    rho: float = 1000.0
    rgba: tuple = WATER_RGBA

    # ------------------------------------------------------------------ fields
    def _inside(self, x, y):
        """Distance (m) from (x, y) to the outside of the union of regions (0 outside)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        best = np.zeros(np.broadcast(x, y).shape)
        for x0, y0, x1, y1 in self.regions:
            d = np.minimum(np.minimum(x - x0, x1 - x), np.minimum(y - y0, y1 - y))
            best = np.maximum(best, d)
        return np.clip(best, 0.0, None)

    def depth_at(self, x, y):
        """Water depth (m) at world (x, y); arrays broadcast."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        d = np.full(np.broadcast(x, y).shape, float(self.depth))
        for b in self.bands:
            c = x if b.get("axis", "y") == "x" else y
            w = _smooth01((c - b["lo"]) / self.blend + 0.5) * _smooth01((b["hi"] - c) / self.blend + 0.5)
            d = d + (float(b["depth"]) - d) * w
        for p in self.pools:
            r2 = (x - p["center"][0]) ** 2 + (y - p["center"][1]) ** 2
            d = d + float(p["depth"]) * np.exp(-r2 / max(float(p["radius"]), 1e-3) ** 2)
        taper = _smooth01(self._inside(x, y) / max(self.edge, 1e-3))
        out = np.clip(d, 0.0, None) * taper
        return float(out) if out.ndim == 0 else out

    def water_top(self, x, y):
        return self.bed_z + self.depth_at(x, y)

    def flow_at(self, x, y, t: float = 0.0) -> np.ndarray:
        """Surface current (vx, vy) at a point: uniform current + radial outflow around fountains."""
        wet = 1.0 if self.depth_at(x, y) > 0.005 else 0.0
        v = np.array(self.flow, float) * wet
        for g in self.geysers:
            dx, dy = x - g.pos[0], y - g.pos[1]
            r = math.hypot(dx, dy)
            if r > 1e-3:
                u = g.outflow * min(g.strength(t), 1.5) * math.exp(-((r - g.radius) / (1.5 * g.radius)) ** 2)
                v += u * np.array([dx / r, dy / r]) * wet
        return v

    def bounds(self):
        a = np.array(self.regions, float)
        return a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max()

    # ------------------------------------------------------------------ (de)serialise
    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_json(cls, s: str) -> "FloodSpec":
        d = json.loads(s)
        d["geysers"] = [Geyser(**g) for g in d.get("geysers", [])]
        d["regions"] = [tuple(r) for r in d.get("regions", [])]
        d["flow"] = tuple(d.get("flow", (0, 0)))
        d["rgba"] = tuple(d.get("rgba", WATER_RGBA))
        return cls(**d)

    @classmethod
    def from_model(cls, model) -> "FloodSpec":
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, SPEC_TEXT)
        if i < 0:
            raise KeyError("scene has no flood spec: call simkit.flood.add_to_scene(sc, spec) in build_env.py")
        a, n = int(model.text_adr[i]), int(model.text_size[i])
        return cls.from_json(bytes(model.text_data[a:a + n - 1]).decode())


def _smooth01(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


# ====================================================================== scene building
def _smooth_noise(rng, shape, cells: float):
    """Smooth value noise in [0, 1] of ``shape`` (H, W), ~``cells`` features along the long side."""
    from PIL import Image

    H, W = shape
    gh = max(2, int(round(cells * H / max(H, W))) + 2)
    gw = max(2, int(round(cells * W / max(H, W))) + 2)
    a = np.asarray(Image.fromarray(rng.random((gh, gw)).astype(np.float32), mode="F").resize((W, H), Image.BICUBIC))
    return (a - a.min()) / (np.ptp(a) + 1e-9)


def _grain(rng, shape, px: int):
    """White noise with grains of ~px pixels (bilinear), so streaks stay visible after mipmapping."""
    from PIL import Image

    H, W = shape
    g = rng.random((max(2, H // px), max(2, W // px))).astype(np.float32)
    return np.asarray(Image.fromarray(g, mode="F").resize((W, H), Image.BILINEAR))


def _lic(noise, ux, uy, steps: int):
    """Line-integral convolution: average ``noise`` along the streamlines of the unit field (ux, uy)
    (pixel units, column = +x, row = +y) -> streaks that follow the flow."""
    H, W = noise.shape
    acc, wsum = noise.copy(), np.ones_like(noise)
    jj, ii = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    for sgn in (1.0, -1.0):
        px, py = jj.copy(), ii.copy()
        for s in range(steps):
            ci, cj = py.astype(np.int32), px.astype(np.int32)
            px += sgn * ux[ci, cj]
            py += sgn * uy[ci, cj]
            np.clip(px, 0, W - 1, out=px)
            np.clip(py, 0, H - 1, out=py)
            w = 1.0 - s / steps
            acc += w * noise[py.astype(np.int32), px.astype(np.int32)]
            wsum += w
    return acc / wsum


def water_map(spec: FloodSpec, path, max_px: int = 2048, seed: int = 0):
    """One non-tiling texture over the whole flood (u = x, v = y over ``spec.bounds()``): muddy water
    whose streaks follow the current and each fountain's outflow (line-integral convolution of noise
    along the flow field), darker where deeper, ragged white foam boiling out around every fountain
    and trailing downstream, broken foam lines along the waterline. Returns the bounds it covers."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = spec.bounds()
    sx, sy = x1 - x0, y1 - y0
    ppm = min(max_px / max(sx, sy), 40.0)                     # pixels per metre (<= 2.5 cm/px)
    W, H = max(64, int(round(sx * ppm))), max(64, int(round(sy * ppm)))
    X, Y = np.meshgrid(x0 + (np.arange(W) + 0.5) / W * sx, y0 + (np.arange(H) + 0.5) / H * sy)  # row = +y
    D = spec.depth_at(X, Y).astype(np.float32)

    # flow field at mean fountain strength (same terms as flow_at), meandering a little
    vx = np.full(X.shape, float(spec.flow[0]), np.float32)
    vy = np.full(X.shape, float(spec.flow[1]), np.float32)
    for g in (g for g in spec.geysers if g.start <= 0):   # a later eruption leaves no trace in the texture
        dx, dy = X - g.pos[0], Y - g.pos[1]
        r = np.hypot(dx, dy) + 1e-6
        u = g.outflow * np.exp(-((r - g.radius) / (1.5 * g.radius)) ** 2)
        vx += (u * dx / r).astype(np.float32)
        vy += (u * dy / r).astype(np.float32)
    speed = np.hypot(vx, vy)
    ang = np.where(speed > 0.03, np.arctan2(vy, vx), 0.0) + 0.5 * (_smooth_noise(rng, X.shape, max(sx, sy) / 4) - 0.5)
    ux, uy = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    steps = int(np.clip(0.9 * ppm, 8, 32))
    noise = 0.7 * _grain(rng, X.shape, 2) + 0.3 * _grain(rng, X.shape, 5)
    s = _lic(noise.astype(np.float32), ux, uy, steps)
    lo, hi = np.percentile(s, [2, 98])
    s = np.clip((s - lo) / (hi - lo + 1e-9), 0, 1)
    patch = _smooth_noise(rng, X.shape, max(sx, sy) / 6)

    base = np.array(spec.rgba[:3], np.float32)
    deep = np.clip(D / 0.45, 0, 1)[..., None]
    col = base * (1.08 - 0.28 * deep) * (0.86 + 0.24 * s[..., None] + 0.12 * (patch[..., None] - 0.5))
    film = np.clip(1 - D / 0.04, 0, 1)[..., None]               # a few cm: the wet road shows through
    col = col * (1 - 0.55 * film) + np.array([0.17, 0.17, 0.18]) * 0.55 * film

    foam = np.zeros(X.shape, np.float32)
    fl = np.array(spec.flow, float)
    fdir = fl / np.linalg.norm(fl) if np.linalg.norm(fl) > 0.05 else None
    for g in (g for g in spec.geysers if g.start <= 0):
        R = g.radius
        dx, dy = X - g.pos[0], Y - g.pos[1]
        r = np.hypot(dx, dy)
        f = np.maximum(np.exp(-(r / (0.95 * R)) ** 2), 0.7 * np.exp(-((r - 1.4 * R) / (0.65 * R)) ** 2))
        if fdir is not None:                                     # foam carried downstream as a widening plume
            along = dx * fdir[0] + dy * fdir[1]
            across = -dx * fdir[1] + dy * fdir[0]
            a = np.clip(along, 0, None)
            plume = 0.8 * np.exp(-(across / (0.8 * R + 0.3 * a)) ** 2) * np.exp(-a / (5 * R)) / (1 + np.exp(-along / (0.4 * R)))
            f = np.maximum(f, plume)
        foam = np.maximum(foam, f.astype(np.float32))
    inside = spec._inside(X, Y)
    foam = np.maximum(foam, 0.55 * np.exp(-((inside - 0.3) / 0.3) ** 2) * (D > 0.01))   # waterline
    rag = 0.6 * s + 0.4 * _smooth_noise(rng, X.shape, max(sx, sy) / 0.8)
    mask = 0.92 * _smooth01((foam - 0.55 * rag - 0.12) / 0.25)
    mask = np.maximum(mask, 0.4 * np.clip((s - 0.86) / 0.12, 0, 1) ** 2 * (D > 0.03))   # thin streak foam
    foam_col = np.array([0.86, 0.85, 0.81]) * (0.86 + 0.18 * s[..., None])
    col = col * (1 - mask[..., None]) + foam_col * mask[..., None]
    Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).save(path)
    return x0, y0, x1, y1


def add_to_scene(sc, spec: FloodSpec, env_dir=".", res: float = 0.5, foam: bool = True, splashes: int = 12,
                 wakes=(), seed: int = 0, max_px: int = 2048):
    """Water surface, fountain boils, vehicle bow waves, and the spec itself (custom text inside
    scene.xml, read back by FloodWorld and the check helpers).

    Water: a gently rippled mesh over the flood (visual only, no collision) with ONE texture baked
    for this spec (``water_map``): streaks that follow the current and the fountains' outflow, darker
    deep water, ragged foam around each fountain and downstream of it, foam lines at the waterline.
    Fountains: a churning cluster of foam lumps that pulses with the jet plus small blobs bobbing
    around it (mocap, animated by FloodWorld). wakes: vehicle names that get a bow wave in water.
    env_dir: the environment folder (scene.xml's folder); the texture goes to
    <env_dir>/assets/procedural/flood_water.png.
    """
    from pathlib import Path

    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = spec.bounds()
    nx = int(np.clip(round((x1 - x0) / res) + 1, 8, 160))
    ny = int(np.clip(round((y1 - y0) / res) + 1, 8, 160))
    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    D = spec.depth_at(X, Y)
    wet = D > 0.004
    ripple = 0.005 * rng.standard_normal(X.shape)  # breaks up the specular highlight; the streaks are in the texture
    Z = np.where(wet, spec.bed_z + D + np.where(D > 0.03, ripple, 0.0), spec.bed_z - 0.05)  # dry vertices tuck under the road
    fx, fy = spec.flow
    ang = math.atan2(fy, fx) if math.hypot(fx, fy) > 0.05 else None
    idx = -np.ones((nx, ny), int)
    verts, uvs, faces = [], [], []

    def vid(i, j):
        if idx[i, j] < 0:
            idx[i, j] = len(verts)
            verts.append((X[i, j], Y[i, j], Z[i, j]))
            uvs.append(((X[i, j] - x0) / (x1 - x0), (Y[i, j] - y0) / (y1 - y0)))   # one texture over the flood
        return idx[i, j]
    for i in range(nx - 1):
        for j in range(ny - 1):
            if wet[i:i + 2, j:j + 2].any():
                a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
                faces += [(a, b, c), (a, c, d)]
    if not faces:
        raise ValueError("flood spec has no water (check regions and depth)")
    tex_dir = Path(env_dir) / "assets" / "procedural"
    tex_dir.mkdir(parents=True, exist_ok=True)
    water_map(spec, tex_dir / "flood_water.png", max_px=max_px, seed=seed)
    rel = "assets/procedural/flood_water.png"
    v = " ".join(f"{p:.3f}" for xyz in verts for p in xyz)
    t = " ".join(f"{p:.5f}" for uv in uvs for p in uv)
    f = " ".join(str(k) for tri in faces for k in tri)
    sc.asset(f'<texture name="flood_water_tex" type="2d" file="{rel}"/>')
    sc.asset(f'<material name="flood_water_mat" texture="flood_water_tex" rgba="1 1 1 {spec.rgba[3]}" specular="0.7" shininess="0.85"/>')
    sc.asset(f'<mesh name="flood_water_mesh" vertex="{v}" texcoord="{t}" face="{f}" inertia="shell"/>')
    sc.body('<geom name="flood_water" type="mesh" mesh="flood_water_mesh" material="flood_water_mat" '
            'contype="0" conaffinity="0" group="2" mass="0"/>')
    fr, fg, fb, _ = FOAM_RGBA
    # matte, speckled foam (a smooth white ellipsoid reads as cotton wool)
    sc.asset(f'<texture name="flood_foam_tex" type="cube" builtin="flat" mark="random" random="0.2" width="64" height="64" '
             f'rgb1="{fr} {fg} {fb}" markrgb="{0.86 * fr:.2f} {0.86 * fg:.2f} {0.84 * fb:.2f}"/>')
    sc.asset('<material name="flood_foam_mat" texture="flood_foam_tex" specular="0.12" shininess="0.08"/>')
    sc.asset(f'<material name="flood_foam_soft" texture="flood_foam_tex" rgba="1 1 1 0.75" specular="0.08" shininess="0.05"/>')
    G_ = 'contype="0" conaffinity="0" group="2" mass="0"'
    for gz in spec.geysers if foam else []:
        top = spec.water_top(*gz.pos)
        R, H = gz.radius, gz.height * (1 + gz.pulse)
        # the boil: a column of foam with lower lumps leaning around it; rises and falls with the jet
        # a low churning mound (a jet h tall falls back into a mound ~2.5 h across) with lumps around it;
        # (dx, dy as fractions of W, horizontal half-sizes as fractions of W, height above the water as a fraction of h)
        h = gz.height
        W = max(0.6 * R, 1.25 * h)
        lumps = [(0.0, 0.0, 0.8, 0.72, 1.0), (0.62, -0.35, 0.45, 0.4, 0.55), (-0.55, 0.42, 0.42, 0.45, 0.45),
                 (0.15, 0.7, 0.38, 0.34, 0.6), (-0.42, -0.6, 0.4, 0.36, 0.35), (0.75, 0.3, 0.3, 0.28, 0.3)]
        g = ""
        for ox, oy, sx_, sy_, q in lumps:
            hz = (q + 0.3) * h if q < 1 else H      # half-height; the main lump is the pulsing column
            rz = (H - h + q * h - hz) if q < 1 else 0.0
            g += (f'<geom type="ellipsoid" size="{sx_ * W:.2f} {sy_ * W:.2f} {hz:.2f}" pos="{ox * W:.2f} {oy * W:.2f} {rz:.2f}" '
                  f'euler="0 0 {rng.uniform(0, 180):.0f}" material="flood_foam_mat" {G_}/>')
        sc.body(f'<body name="flood_jet_{gz.name}" mocap="true" pos="{gz.pos[0]:.3f} {gz.pos[1]:.3f} '
                f'{top - H + gz.height:.3f}">{g}</body>')
        for k in range(splashes):   # churning blobs at the surface, half of them carried downstream
            if ang is not None and k % 2:
                a_k = ang + rng.normal(0, 0.8)
                rr = R * rng.uniform(0.6, 1.5)
            else:
                a_k = 2 * math.pi * k / splashes + 0.4 * gz.phase + rng.uniform(-0.3, 0.3)
                rr = R * rng.uniform(0.4, 1.0)
            sz = R * rng.uniform(0.1, 0.2)
            sc.body(f'<body name="flood_jet_{gz.name}_s{k}" mocap="true" pos="{gz.pos[0] + rr * math.cos(a_k):.3f} '
                    f'{gz.pos[1] + rr * math.sin(a_k):.3f} {top:.3f}"><geom type="ellipsoid" '
                    f'size="{1.4 * sz:.2f} {sz:.2f} {0.35 * sz:.2f}" euler="0 0 {math.degrees(a_k):.0f}" '
                    f'material="flood_foam_soft" {G_}/></body>')
    for n in wakes:  # bow wave: a flat white crescent pushed ahead of the bumper (hidden below the water when dry/slow)
        sc.body(f'<body name="flood_wake_{n}" mocap="true" pos="0 0 -5">'
                f'<geom type="ellipsoid" size="0.5 1.25 0.16" material="flood_foam_mat" {G_}/>'
                f'<geom type="ellipsoid" size="0.9 0.3 0.08" pos="-1.2 1.05 0" euler="0 0 -18" material="flood_foam_soft" {G_}/>'
                f'<geom type="ellipsoid" size="0.9 0.3 0.08" pos="-1.2 -1.05 0" euler="0 0 18" material="flood_foam_soft" {G_}/>'
                f'</body>')
    sc.custom(f'<text name="{SPEC_TEXT}" data="{_xml_attr(spec.to_json())}"/>')
    return sc


def storm_look(sc, fog_start: float = 0.25, fog_end: float = 1.6, fog_rgba=(0.62, 0.64, 0.66, 1.0)):
    """Overcast storm: grey sky, soft shadowless light, haze and distance fog (fractions of the scene
    extent ~ its radius). Use INSTEAD of sc.sky(...). Fog is drawn by simkit renderers (smoke,
    showcase, reference compare); the native viewer shows it too."""
    sc.sky("storm")
    r, g, b, a = fog_rgba
    sc.set_visual("map", fogstart=fog_start, fogend=fog_end)
    sc.set_visual("rgba", fog=f"{r} {g} {b} {a}", haze=f"{r} {g} {b} 1")
    sc.custom('<numeric name="simkit_fog" data="1"/>')
    return sc


def _xml_attr(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


# ====================================================================== physics
def _name(model, obj, i):
    return mujoco.mj_id2name(model, obj, i) or ""


def find_vehicles(model) -> list[str]:
    """Bodies built with simkit.robots.car (or the same naming): free joint <n>_root + geom <n>_chassis."""
    out = []
    for b in range(1, model.nbody):
        n = _name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if (n and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{n}_chassis") >= 0
                and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_root") >= 0):
            out.append(n)
    return out


_DRY: dict = {}


def _dry_friction(model):
    """Tyre friction as built (FloodWorld lowers model.geom_friction in water; a second FloodWorld on the
    same model must not take the wet value for the dry one)."""
    key = (id(model), int(model.ngeom))
    if key not in _DRY:
        _DRY[key] = np.array(model.geom_friction[:, 0], float)
    return _DRY[key]


class _Vehicle:
    def __init__(self, model, name, clearance=None, hull_height=1.0, fill=0.65, cd=1.0, blockage=0.35):
        self.name = name
        self.bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        cg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_chassis")
        hx, hy, hz = model.geom_size[cg]
        cp = model.geom_pos[cg]
        self.L, self.W = 2 * hx / 0.95, 2 * hy / 0.95       # simkit car: chassis box is 95 % of the body
        self.cx, self.cy = float(cp[0]), float(cp[1])
        self.zb = float(clearance if clearance is not None else min(cp[2] - hz, 0.22))  # underside (body frame); cars ~0.15-0.22 m
        self.H, self.fill, self.cd, self.blockage = hull_height, fill, cd, blockage
        self.tires = [g for g in range(model.ngeom) if _name(model, mujoco.mjtObj.mjOBJ_GEOM, g).startswith(f"{name}_")
                      and _name(model, mujoco.mjtObj.mjOBJ_GEOM, g).endswith("_tire")]
        dry = _dry_friction(model)
        self.mu_dry = {g: float(dry[g]) for g in self.tires}


class FloodWorld:
    """World dynamics of a flood: call it before every mj_step (world.py). Stateless per step, so it is
    safe across resets and for several MjData. ``info[name]`` holds the last depth and forces per vehicle.

    vehicles: names (default: every simkit car found), or {name: {"clearance": m, "hull_height": m,
    "fill": 0.65, "cd": 1.0}} to override per vehicle. ``geysers=False`` switches fountain forces off
    (for A/B checks); ``grip=False`` keeps dry tyre friction.
    """

    def __init__(self, model, spec: FloodSpec | None = None, vehicles=None, geysers: bool = True, grip: bool = True,
                 animate: bool = True):
        self.spec = spec or FloodSpec.from_model(model)
        if vehicles is None:
            vehicles = find_vehicles(model)
        if isinstance(vehicles, dict):
            self.vehicles = [_Vehicle(model, n, **(kw or {})) for n, kw in vehicles.items()]
        else:
            self.vehicles = [_Vehicle(model, n) for n in vehicles]
        self.geysers_on, self.grip_on, self.animate = geysers, grip, animate
        self.info: dict[str, dict] = {}
        self._vel = np.zeros(6)
        self.jets = []
        for gz in self.spec.geysers:
            dome = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"flood_jet_{gz.name}")
            blobs = []
            k = 0
            while True:
                b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"flood_jet_{gz.name}_s{k}")
                if b < 0:
                    break
                blobs.append((int(model.body_mocapid[b]), np.array(model.body_pos[b], float), k))
                k += 1
            self.jets.append((gz, int(model.body_mocapid[dome]) if dome >= 0 else -1, blobs))
        self.wakes = {}
        for v in self.vehicles:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"flood_wake_{v.name}")
            if b >= 0:
                self.wakes[v.name] = int(model.body_mocapid[b])

    def __call__(self, model, data):
        t = float(data.time)
        sp = self.spec
        if self.animate:
            self._animate(data, t)
        for v in self.vehicles:
            self._vehicle(model, data, v, t, sp)

    # ------------------------------------------------------------------
    def _animate(self, data, t):
        sp = self.spec
        for gz, mid, blobs in self.jets:
            top = sp.water_top(*gz.pos)
            s = gz.strength(t)
            H = gz.height * (1 + gz.pulse)
            sink = max(0.0, 1.0 - s / 0.3)   # dormant (before ``start``): everything below the surface
            if mid >= 0:
                data.mocap_pos[mid] = (gz.pos[0], gz.pos[1], top - H + gz.height * s - H * sink)
            for bid, p0, k in blobs:
                ph = 2 * math.pi * t / (0.55 * gz.period) + 1.7 * k + gz.phase
                rise = 0.08 * gz.height * s * (0.5 + 0.5 * math.sin(ph))   # bob at the surface, never float above it
                spread = 1.0 + 0.25 * (0.5 + 0.5 * math.sin(ph + 1.0))
                dx, dy = p0[0] - gz.pos[0], p0[1] - gz.pos[1]
                data.mocap_pos[bid] = (gz.pos[0] + spread * dx, gz.pos[1] + spread * dy, top - 0.03 + rise - 0.4 * sink)

    def _vehicle(self, model, data, v: _Vehicle, t, sp):
        R = data.xmat[v.bid].reshape(3, 3)
        o = data.xpos[v.bid]
        com = data.xipos[v.bid]
        F = np.zeros(3)
        T = np.zeros(3)

        def add(force, point):
            nonlocal F, T
            F += force
            T += np.cross(point - com, force)
        # buoyancy per quarter of the footprint
        q = []
        for sx in (-0.25, 0.25):
            for sy in (-0.25, 0.25):
                p = o + R @ np.array([v.cx + sx * v.L, v.cy + sy * v.W, v.zb])
                top = sp.water_top(p[0], p[1])
                h = float(np.clip(top - p[2], 0.0, v.H))
                q.append(h)
                if h > 0:
                    add(np.array([0, 0, sp.rho * G * v.fill * (v.L * v.W / 4) * h]), p + np.array([0, 0, h / 2]))
        # drag against the water (horizontal), wheel-level and hull-level immersed area
        c = o + R @ np.array([v.cx, v.cy, v.zb])
        depth_c = sp.depth_at(c[0], c[1])
        top_c = sp.bed_z + depth_c
        imm_low = float(np.clip(min(top_c, c[2]) - sp.bed_z, 0.0, None))      # below the underside: wheels
        imm_high = float(np.clip(top_c - c[2], 0.0, v.H))                     # against the hull
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, v.bid, self._vel, 0)
        lin, ang = self._vel[3:].copy(), self._vel[:3].copy()
        vrel = lin[:2] - sp.flow_at(c[0], c[1], t)
        sp_rel = float(np.linalg.norm(vrel))
        drag = np.zeros(3)
        if sp_rel > 1e-4 and (imm_low > 0 or imm_high > 0):
            fwd = R[:2, 0] / (np.linalg.norm(R[:2, 0]) + 1e-9)
            cos = abs(float(vrel @ fwd)) / sp_rel
            sin = math.sqrt(max(0.0, 1 - cos * cos))
            width = v.W * cos + v.L * sin
            A = width * (v.blockage * imm_low + imm_high)
            drag[:2] = -0.5 * sp.rho * v.cd * A * sp_rel * vrel
            add(drag, np.array([c[0], c[1], sp.bed_z + 0.5 * (imm_low + imm_high)]))
        # heave / roll / pitch damping while the hull is wet (water resists vertical motion)
        wet = float(np.mean([h > 0 for h in q]))
        if wet > 0:
            F[2] += -0.5 * sp.rho * 1.0 * (v.L * v.W) * wet * abs(lin[2]) * lin[2]
            T += -wet * 4000.0 * np.array([ang[0], ang[1], 0.2 * ang[2]])
        # manhole fountains under the vehicle
        jet = 0.0
        if self.geysers_on:
            for gz in sp.geysers:
                local = R.T @ (np.array([gz.pos[0], gz.pos[1], 0.0]) - o)
                if abs(local[0] - v.cx) < v.L / 2 and abs(local[1] - v.cy) < v.W / 2:
                    p = o + R @ np.array([local[0], local[1], v.zb])
                    top = sp.water_top(gz.pos[0], gz.pos[1])
                    h = gz.height * gz.strength(t)
                    reach = float(np.clip(1.0 - (p[2] - top) / max(h, 1e-3), 0.0, 1.0))
                    fj = gz.jet_force(t, sp.rho) * reach
                    if fj > 0:
                        add(np.array([0, 0, fj]), p)
                        jet += fj
        # tyre grip in water
        if self.grip_on:
            for g in v.tires:
                p = data.geom_xpos[g]
                d = sp.depth_at(p[0], p[1])
                model.geom_friction[g, 0] = v.mu_dry[g] * (1.0 - 0.35 * min(d / 0.3, 1.0))
        data.xfrc_applied[v.bid, :3] = F
        data.xfrc_applied[v.bid, 3:] = T
        mid = self.wakes.get(v.name, -1)
        if mid >= 0 and self.animate:  # bow wave grows with speed through the water and with depth
            fwd = R[:, 0]
            yaw = math.atan2(fwd[1], fwd[0])
            front = o + R @ np.array([v.cx + v.L / 2 + 0.35, v.cy, 0.0])
            d_front = sp.depth_at(front[0], front[1])
            spd = float(np.linalg.norm(lin[:2] - sp.flow_at(front[0], front[1], t)))
            k = min(spd / 5.0, 1.0) * min(d_front / 0.25, 1.0)
            z = sp.bed_z + d_front - 0.2 + 0.3 * k if k > 0.05 else -5.0
            data.mocap_pos[mid] = (front[0], front[1], z)
            data.mocap_quat[mid] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.info[v.name] = {"depth": round(float(depth_c), 3), "submerged_hull_m": round(float(np.mean(q)), 3),
                             "buoyancy_N": round(float(sp.rho * G * v.fill * (v.L * v.W / 4) * sum(q)), 1),
                             "drag_N": round(float(np.linalg.norm(drag)), 1), "jet_N": round(jet, 1)}


# ====================================================================== helpers for task checks
def wheel_load(model, data, name: str) -> float:
    """Total normal contact force (N) on the tyres of vehicle ``name`` (geoms <name>_*_tire)."""
    tires = {g for g in range(model.ngeom) if _name(model, mujoco.mjtObj.mjOBJ_GEOM, g).startswith(f"{name}_")
             and _name(model, mujoco.mjtObj.mjOBJ_GEOM, g).endswith("_tire")}
    f6 = np.zeros(6)
    tot = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in tires or c.geom2 in tires:
            mujoco.mj_contactForce(model, data, i, f6)
            tot += abs(f6[0])
    return tot


def find_spot(spec: FloodSpec, depth: float, near=None, avoid_geysers: float = 2.5, step: float = 0.25):
    """A point (x, y) in the flood whose depth is closest to ``depth`` (nearest to ``near`` among
    equally good ones), away from fountains."""
    x0, y0, x1, y1 = spec.bounds()
    X, Y = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step), indexing="ij")
    D = spec.depth_at(X, Y)
    cost = np.abs(D - depth)
    for g in spec.geysers:
        cost = np.where((X - g.pos[0]) ** 2 + (Y - g.pos[1]) ** 2 < avoid_geysers ** 2, np.inf, cost)
    if near is not None:
        cost = cost + 1e-3 * np.hypot(X - near[0], Y - near[1])
    i = np.unravel_index(int(np.argmin(cost)), cost.shape)
    return float(X[i]), float(Y[i]), float(D[i])


# ====================================================================== driving helpers
def pass_fountain(spec: FloodSpec, gz: "Geyser", route_pt, heading, *, half_width: float = 1.0,
                  safe_depth: float = 0.3, margin: float = 0.8, lateral_limits=(-math.inf, math.inf),
                  obstacles=(), movers=(), speed: float = 3.0, approach: float = 10.0) -> dict | None:
    """Plan to drive PAST a fountain that sits on a straight route instead of stopping in front of it
    (a car stopped in the water next to an erupting manhole is the failure this avoids).

    route_pt, heading: a point on the route (lane centre line) and its direction. lateral_limits:
    (min, max) lateral offset of the car CENTRE LINE's edges allowed, relative to the route, + = left
    of heading (kerbs, the centre line you may not cross). obstacles: [(x, y, radius)] parked cars.
    movers: [(x, y, vx, vy, radius)] moving cars, checked in time: route_pt is then the vehicle's
    current position (on its lane line) and ``speed`` its speed through the swerve; a car that will
    have left by the time we get there does not block. Returns None when the fountain is not in the
    way; else
    {"offset": lateral offset to drive at while passing (m, + = left), "side": "left"|"right",
     "s_fountain": along-route distance of the fountain from route_pt, "window": (s0, s1) where the
     swerve starts/ends, "keep": clearance radius, "max_depth": deepest water under the car on the
     detour, "candidates": [...]}; "offset" is None when neither side is passable (then stop BEFORE
    the water, never inside it). Feed the plan to ``swerve_path`` for waypoints."""
    h = np.asarray(heading, float) / np.linalg.norm(heading)
    n = np.array([-h[1], h[0]])
    p0 = np.asarray(route_pt, float)[:2]
    d = np.asarray(gz.pos, float)[:2] - p0
    s_g, e = float(d @ h), float(d @ n)
    keep = gz.radius + half_width + margin
    if abs(e) >= keep:
        return None
    plan = {"s_fountain": s_g, "keep": keep, "window": (s_g - keep - approach, s_g + keep + approach)}
    cands = []
    for side, off in (("left", e + keep), ("right", e - keep)):
        c = {"side": side, "offset": round(off, 3)}
        lo, hi = lateral_limits
        if off - half_width < lo or off + half_width > hi:
            c["blocked"] = "road edge"
            cands.append(c)
            continue
        ss = np.arange(plan["window"][0], plan["window"][1] + 0.01, 0.5)
        prof = off * _swerve_profile(ss, s_g, keep, approach)
        worst, hit = 0.0, None
        for w in (-half_width, 0.0, half_width):
            pts = p0[None, :] + ss[:, None] * h[None, :] + (prof + w)[:, None] * n[None, :]
            worst = max(worst, float(np.max(spec.depth_at(pts[:, 0], pts[:, 1]))))
            for ox, oy, rr in obstacles:
                if np.min(np.hypot(pts[:, 0] - ox, pts[:, 1] - oy)) < rr + 0.3:
                    hit = (round(ox, 1), round(oy, 1))
            tau = np.clip(ss, 0, None) / max(speed, 0.5)                  # when we are at each sample
            for mx, my, mvx, mvy, rr in movers:
                gap = np.hypot(pts[:, 0] - (mx + mvx * tau), pts[:, 1] - (my + mvy * tau))
                if np.min(gap) < rr + 0.5:
                    hit = (round(mx, 1), round(my, 1), "moving")
        c["max_depth"] = round(worst, 3)
        if hit:
            c["blocked"] = f"obstacle at {hit}"
        elif worst >= safe_depth:
            c["blocked"] = f"too deep ({worst:.2f} m >= {safe_depth} m)"
        cands.append(c)
    ok = [c for c in cands if "blocked" not in c]
    best = min(ok, key=lambda c: (round(c["max_depth"], 2), abs(c["offset"]))) if ok else None
    plan.update(candidates=cands, offset=best["offset"] if best else None, side=best["side"] if best else None,
                max_depth=best["max_depth"] if best else None)
    return plan


def _swerve_profile(s, s_g, keep, approach):
    """0 -> 1 (smoothstep) over ``approach`` metres before the fountain's clearance circle, 1 while
    passing it, back to 0 over ``approach`` metres after."""
    s = np.asarray(s, float)
    up = _smooth01((s - (s_g - keep - approach)) / approach)
    down = _smooth01(((s_g + keep + approach) - s) / approach)
    return np.minimum(up, down)


def swerve_path(route_pt, heading, plan: dict, s_from: float, s_to: float, step: float = 2.0):
    """Waypoints [(x, y)] along the route from s_from to s_to (along-route distances from route_pt)
    with the swerve of a ``pass_fountain`` plan applied."""
    h = np.asarray(heading, float) / np.linalg.norm(heading)
    n = np.array([-h[1], h[0]])
    p0 = np.asarray(route_pt, float)[:2]
    ss = np.arange(s_from, s_to + 1e-6, step)
    off = plan["offset"] * _swerve_profile(ss, plan["s_fountain"], plan["keep"], plan["window"][1] - plan["s_fountain"] - plan["keep"])
    pts = p0[None, :] + ss[:, None] * h[None, :] + off[:, None] * n[None, :]
    return [(float(x), float(y)) for x, y in pts]


def place_vehicle(model, data, name: str, x: float, y: float, yaw_deg: float, speed: float = 0.0):
    """Teleport a free-joint vehicle (simkit car) to (x, y) at ground level, heading yaw, rolling at
    ``speed`` m/s (wheel spin set to match, so it neither skids nor jumps)."""
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_root")
    qa, da = model.jnt_qposadr[j], model.jnt_dofadr[j]
    z = float(data.qpos[qa + 2])
    q = math.radians(yaw_deg)
    data.qpos[qa:qa + 7] = (x, y, z, math.cos(q / 2), 0, 0, math.sin(q / 2))
    data.qvel[da:da + 6] = (speed * math.cos(q), speed * math.sin(q), 0, 0, 0, 0)
    for t in ("fl", "fr", "rl", "rr"):
        sj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_{t}_spin")
        if sj >= 0:
            tire = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_{t}_tire")
            r = float(model.geom_size[tire, 0]) if tire >= 0 else 0.33
            data.qvel[model.jnt_dofadr[sj]] = speed / r
    mujoco.mj_forward(model, data)


def _fresh(model, keyframe="home"):
    d = mujoco.MjData(model)
    k = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if k >= 0:
        mujoco.mj_resetDataKeyframe(model, d, k)
    mujoco.mj_forward(model, d)
    return d


def _speed(model, data, name):
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
    return float(np.linalg.norm(v[3:5]))


def coast_test(model, world, name: str, xy, yaw_deg: float, v0: float = 5.0, seconds: float = 1.5, extra=None) -> dict:
    """Roll vehicle ``name`` at ``v0`` from ``xy`` with zero throttle/brake through the water and measure
    the mean deceleration (m/s^2). ``world`` = callable(model, data) run before each step (FloodWorld
    or the env's world); ``extra`` = optional callable(model, data) (e.g. hold other cars still)."""
    d = _fresh(model)
    place_vehicle(model, d, name, xy[0], xy[1], yaw_deg, v0)
    d.ctrl[:] = 0
    n = int(seconds / model.opt.timestep)
    for _ in range(n):
        world(model, d)
        if extra:
            extra(model, d)
        mujoco.mj_step(model, d)
    v1 = _speed(model, d, name)
    return {"v0": v0, "v1": round(v1, 3), "decel": round((v0 - v1) / seconds, 3)}


def fountain_uplift(model, name: str, geyser: str | None = None, seconds: float = 1.2, spec: FloodSpec | None = None) -> dict:
    """Park vehicle ``name`` over a fountain and compare its tyre load with fountains on vs off: the
    drop is the upward force the jet puts on the car (N). Also reports the jet's predicted force."""
    spec = spec or FloodSpec.from_model(model)
    gz = next((g for g in spec.geysers if geyser in (None, g.name)), None)
    if gz is None:
        raise KeyError(f"no fountain {geyser!r}")
    loads = {}
    for on in (False, True):
        w = FloodWorld(model, spec, geysers=on)
        d = _fresh(model)
        place_vehicle(model, d, name, gz.pos[0], gz.pos[1], 0.0, 0.0)
        acc, cnt, pred = 0.0, 0, 0.0
        n = int(seconds / model.opt.timestep)
        for i in range(n):
            w(model, d)
            mujoco.mj_step(model, d)
            if i > n // 2:
                acc += wheel_load(model, d, name)
                pred += w.info.get(name, {}).get("jet_N", 0.0)
                cnt += 1
        loads[on] = (acc / max(cnt, 1), pred / max(cnt, 1))
    drop = loads[False][0] - loads[True][0]
    return {"wheel_load_off_N": round(loads[False][0], 1), "wheel_load_on_N": round(loads[True][0], 1),
            "uplift_N": round(drop, 1), "jet_force_predicted_N": round(loads[True][1], 1)}
