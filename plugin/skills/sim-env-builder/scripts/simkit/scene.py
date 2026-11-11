"""Small MJCF scene builder with sane defaults. You still write bodies as XML strings.

    from simkit.scene import Scene, lookat
    sc = Scene("kitchen", timestep=0.002)
    sc.ground(size=20, style="tiles")             # plane + texture; style: grid|tiles|asphalt|grass|sand|concrete|wood|water|moon
    sc.sky("day")                                  # day|dusk|night|studio|space
    sc.add_asset(mug_asset)                        # simkit.assets.MjAsset or raw XML string
    sc.body('<body name="table" pos="0 0 0">...</body>')
    sc.camera("overview", pos=(3, -3, 2.5), target=(0, 0, 0.8))
    sc.actuator('<motor name="m1" joint="j1" gear="1" ctrlrange="-1 1"/>')
    sc.write("scene.xml")                         # also compiles it once to catch errors early

Every section (default, asset, worldbody, contact, equality, tendon, actuator, sensor, keyframe)
accepts raw XML so nothing in MJCF is out of reach.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

GROUND_STYLES = {
    # builtin, rgb1, rgb2, mark, texrepeat-per-metre, reflectance
    "grid": ("checker", ".32 .34 .38", ".26 .28 .32", "edge", 1.0, 0.05),
    "tiles": ("checker", ".80 .80 .78", ".70 .70 .68", "edge", 2.0, 0.08),
    "asphalt": ("flat", ".22 .22 .24", ".26 .26 .28", "random", 0.5, 0.0),
    "grass": ("flat", ".30 .48 .22", ".26 .42 .18", "random", 0.5, 0.0),
    "sand": ("flat", ".76 .68 .50", ".70 .62 .45", "random", 0.5, 0.0),
    "concrete": ("flat", ".58 .58 .56", ".52 .52 .50", "random", 0.5, 0.0),
    "wood": ("checker", ".55 .38 .22", ".50 .34 .19", "none", 1.0, 0.05),
    "water": ("flat", ".10 .32 .48", ".12 .36 .52", "random", 0.3, 0.3),
    "moon": ("flat", ".52 .51 .49", ".46 .45 .43", "random", 0.5, 0.0),
}
SKIES = {
    "day": ("gradient", ".55 .72 .92", ".95 .97 1.0", 0.35, ".5 .5 .5"),
    "dusk": ("gradient", ".30 .25 .45", ".95 .62 .40", 0.3, ".5 .45 .4"),
    "night": ("gradient", ".02 .03 .08", ".08 .10 .18", 0.15, ".25 .25 .3"),
    "studio": ("gradient", ".78 .80 .84", ".95 .95 .96", 0.4, ".6 .6 .6"),
    "space": ("gradient", "0 0 0", ".02 .02 .05", 0.1, ".7 .7 .7"),
}


def lookat(pos, target, up=(0.0, 0.0, 1.0)) -> str:
    """MJCF ``xyaxes`` for a camera at ``pos`` looking at ``target`` (MuJoCo cameras look along -z)."""
    p, t, u = (np.asarray(x, float) for x in (pos, target, up))
    fwd = t - p
    fwd /= np.linalg.norm(fwd) + 1e-12
    if abs(float(np.dot(fwd, u))) > 0.999:  # looking straight down/up: pick another up
        u = np.array([0.0, 1.0, 0.0])
    x = np.cross(fwd, u)
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(x, fwd)
    return " ".join(f"{c:.4f}" for c in (*x, *y))


def euler_to_quat(rx, ry, rz, degrees=True) -> tuple[float, float, float, float]:
    """MuJoCo default euler sequence 'xyz' (intrinsic)."""
    if degrees:
        rx, ry, rz = (math.radians(a) for a in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(rx / 2), math.sin(rx / 2), math.cos(ry / 2), math.sin(ry / 2), math.cos(rz / 2), math.sin(rz / 2)
    w = cx * cy * cz - sx * sy * sz
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    return (w, x, y, z)


class Scene:
    def __init__(self, name: str, timestep: float = 0.002, gravity=(0, 0, -9.81), integrator: str = "implicitfast",
                 cone: str = "pyramidal", iterations: int = 50, density: float = 0.0, viscosity: float = 0.0,
                 wind=(0, 0, 0), offscreen=(1600, 900), extent: float | None = None, option_extra: str = ""):
        self.name = name
        self.option = (f'<option timestep="{timestep}" gravity="{gravity[0]} {gravity[1]} {gravity[2]}" '
                       f'integrator="{integrator}" cone="{cone}" iterations="{iterations}" '
                       f'density="{density}" viscosity="{viscosity}" wind="{wind[0]} {wind[1]} {wind[2]}" {option_extra}/>')
        self.offscreen = offscreen
        self.extent = extent
        self.sections: dict[str, list[str]] = {k: [] for k in (
            "default", "asset", "worldbody", "contact", "equality", "tendon", "actuator", "sensor", "keyframe")}
        self._asset_names: set[str] = set()
        self.home: dict[str, float] = {}  # joint name -> value for the auto "home" keyframe
        self.compiler = '<compiler angle="degree" autolimits="true"/>'
        self.size_xml = ""
        self.visual_extra = ""  # legacy: raw <tag .../> strings, merged by set_visual rules
        self._visual: dict[str, dict] = {}

    # -- environment dressing ---------------------------------------------------------
    def sky(self, kind: str = "day", headlight: bool = True):
        builtin, rgb1, rgb2, haze, amb = SKIES[kind]
        self.asset(f'<texture name="skybox" type="skybox" builtin="{builtin}" rgb1="{rgb2}" rgb2="{rgb1}" width="512" height="3072"/>')
        if headlight:
            self.set_visual("headlight", ambient=amb, diffuse=".35 .35 .35", specular=".1 .1 .1")
        else:
            self.set_visual("headlight", active="0")
        self.set_visual("rgba", haze=f".8 .85 .9 {haze}")
        if kind != "space":
            self.body('<light name="sun" directional="true" pos="0 0 50" dir="-0.35 -0.25 -1" castshadow="true" diffuse=".75 .72 .68" specular=".2 .2 .2"/>')
        else:
            self.body('<light name="sun" directional="true" pos="0 0 50" dir="-0.6 -0.2 -0.75" castshadow="true" diffuse="1 1 .98"/>')
        return self

    def ground(self, size: float = 20.0, style: str = "grid", friction: str = "1 0.005 0.0001", z: float = 0.0,
               name: str = "ground", condim: int = 3):
        builtin, rgb1, rgb2, mark, rep, refl = GROUND_STYLES[style]
        extra = ' markrgb=".45 .47 .5"' if mark == "edge" else (' markrgb="0.3 0.3 0.3" random="0.03"' if mark == "random" else "")
        self.asset(f'<texture name="{name}_tex" type="2d" builtin="{builtin}" rgb1="{rgb1}" rgb2="{rgb2}" '
                   f'mark="{mark}"{extra} width="512" height="512"/>')
        self.asset(f'<material name="{name}_mat" texture="{name}_tex" texrepeat="{size * rep:g} {size * rep:g}" reflectance="{refl}"/>')
        self.body(f'<geom name="{name}" type="plane" size="{size} {size} 0.1" pos="0 0 {z}" material="{name}_mat" '
                  f'friction="{friction}" condim="{condim}"/>')
        return self

    # -- raw sections ---------------------------------------------------------------
    def asset(self, xml: str):
        self.sections["asset"].append(xml)
        return self

    def add_asset(self, a):
        """Register an ``MjAsset`` (from simkit.assets / simkit.procedural) once, or raw XML."""
        if isinstance(a, str):
            return self.asset(a)
        if a.name in self._asset_names:
            return self
        self._asset_names.add(a.name)
        return self.asset(a.asset_xml())

    def body(self, xml: str):
        self.sections["worldbody"].append(xml)
        return self

    def default(self, xml: str):
        self.sections["default"].append(xml)
        return self

    def contact(self, xml: str):
        self.sections["contact"].append(xml)
        return self

    def equality(self, xml: str):
        self.sections["equality"].append(xml)
        return self

    def tendon(self, xml: str):
        self.sections["tendon"].append(xml)
        return self

    def actuator(self, xml: str):
        self.sections["actuator"].append(xml)
        return self

    def sensor(self, xml: str):
        self.sections["sensor"].append(xml)
        return self

    def keyframe(self, xml: str):
        self.sections["keyframe"].append(xml)
        return self

    def camera(self, name: str, pos, target, fovy: float = 50.0, body: str | None = None, mode: str = "fixed"):
        """Fixed world camera looking at ``target``. For a camera riding on a robot, put a
        ``<camera>`` inside that robot's body instead (use ``lookat`` for its xyaxes)."""
        xml = (f'<camera name="{name}" pos="{pos[0]:g} {pos[1]:g} {pos[2]:g}" xyaxes="{lookat(pos, target)}" '
               f'fovy="{fovy:g}" mode="{mode}"/>')
        if body is None:
            self.body(xml)
        return xml

    # -- output ----------------------------------------------------------------------
    def set_visual(self, element: str, **attrs):
        """Set attributes of a <visual> child once, e.g. ``sc.set_visual("map", znear=0.05, zfar=5000)``,
        ``sc.set_visual("headlight", ambient=".4 .4 .4")``, ``sc.set_visual("quality", shadowsize=8192)``.
        Later calls merge into earlier ones, so no element is ever written twice."""
        self._visual.setdefault(element, {}).update({k.rstrip("_"): v for k, v in attrs.items()})
        return self

    def _visual_xml(self) -> str:
        import re

        w, h = self.offscreen
        merged = {"global": {"offwidth": w, "offheight": h}, "quality": {"shadowsize": 4096, "offsamples": 8},
                  "map": {"znear": 0.01, "zfar": 2000}}
        for el, attrs in self._visual.items():
            merged.setdefault(el, {}).update(attrs)
        # anything written straight into visual_extra (<tag a="b"/>) is merged too, never duplicated
        for tag, body in re.findall(r"<(\w+)([^>]*)/>", self.visual_extra):
            merged.setdefault(tag, {}).update(dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', body)))
        inner = "".join(f"<{el} " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "/>" for el, attrs in merged.items())
        return f"<visual>{inner}</visual>"

    def to_xml(self) -> str:
        ext = f' extent="{self.extent}"' if self.extent else ""
        visual = self._visual_xml()
        parts = [f'<mujoco model="{self.name}">', self.compiler, self.option, self.size_xml, visual,
                 f"<statistic{ext} center=\"0 0 0\"/>" if ext else ""]
        for sec, items in self.sections.items():
            if items:
                parts.append(f"<{sec}>\n" + "\n".join(items) + f"\n</{sec}>")
        parts.append("</mujoco>")
        return "\n".join(p for p in parts if p)

    def write(self, path: str | Path, check: bool = True) -> Path:
        """Write the scene. If robots registered home poses (``self.home``) and no keyframe was
        added by hand, a ``home`` keyframe is generated by joint name: qpos from the bodies'
        placed poses + home joint values, and ctrl = home value for position actuators on those
        joints (so reset() starts with the robot holding its pose)."""
        import mujoco

        path = Path(path)
        path.write_text(_pretty(self.to_xml()))
        if self.home and not self.sections["keyframe"]:
            m = mujoco.MjModel.from_xml_path(str(path))
            qpos = m.qpos0.copy()
            ctrl = np.zeros(m.nu)
            missing = []
            for jn, val in self.home.items():
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
                if j < 0:
                    missing.append(jn)
                    continue
                qpos[m.jnt_qposadr[j]] = val
            for a in range(m.nu):
                j = int(m.actuator_trnid[a, 0])
                if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) and m.jnt_type[j] in (2, 3):
                    is_pos = m.actuator_biastype[a] == 1 and abs(m.actuator_biasprm[a, 1] + m.actuator_gainprm[a, 0]) < 1e-9
                    if is_pos:
                        ctrl[a] = qpos[m.jnt_qposadr[j]]
            if missing:
                raise KeyError(f"home pose names unknown joints: {missing}")
            self.sections["keyframe"].append(
                f'<key name="home" qpos="{" ".join(f"{x:.6g}" for x in qpos)}"' + (f' ctrl="{" ".join(f"{x:.6g}" for x in ctrl)}"' if m.nu else "") + "/>")
            path.write_text(_pretty(self.to_xml()))
            self.sections["keyframe"].pop()
        if check:
            mujoco.MjModel.from_xml_path(str(path))  # raises with a line number if invalid
        return path


def _pretty(xml: str) -> str:
    try:
        import xml.dom.minidom as md

        txt = md.parseString(xml).toprettyxml(indent="  ")
        return "\n".join(line for line in txt.splitlines() if line.strip() and not line.startswith("<?xml"))
    except Exception:  # noqa: BLE001 - keep raw text if it is not well-formed; MuJoCo will report
        return xml
